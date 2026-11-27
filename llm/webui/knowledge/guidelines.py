"""The original variant-interpretation guidelines, downloaded and searchable, so the model
answers rule questions from the published text (with a citation) instead of from memory.

    python -m webui.knowledge.guidelines fetch     # download (internet; a few minutes)
    python -m webui.knowledge.guidelines build     # split into passages + full-text index
    python -m webui.knowledge.guidelines all       # both

Sources (all public / open access):
  - ACMG/AMP 2015 and the ClinGen SVI recommendation papers, full text from PubMed Central
    (BioC JSON), including their tables.
  - Every RELEASED ClinGen Variant Curation Expert Panel (VCEP) specification from the
    ClinGen criteria-specification registry (cspec.genome.network): per gene, per criterion,
    per strength, the panel's exact wording.

Files: llm/data/guidelines/raw/ (downloads) and llm/data/guidelines.sqlite (index), both
git-ignored. Re-run `all` to refresh.
"""
from __future__ import annotations

import html
import json
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

import httpx

from ..config import settings

DATA = settings.genetics_db.parent
RAW = DATA / "guidelines" / "raw"
INDEX = DATA / "guidelines.sqlite"
UA = {"User-Agent": "advat-llm-chat/1.0"}
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
CSPEC_LIST = "https://cspec.genome.network/cspec/api/svis"
CSPEC_DOC = "https://cspec.genome.network/cspec/api/SequenceVariantInterpretation/id/{}"

# (PMC id, short citation, what it covers)
PAPERS = [
    ("PMC4544753", "Richards et al. 2015, Genet Med (ACMG/AMP standards)",
     "the 28 ACMG/AMP criteria and combining rules"),
    ("PMC6185798", "Abou Tayoun et al. 2018, Hum Mutat (ClinGen SVI PVS1)",
     "PVS1 decision tree"),
    ("PMC9748256", "Pejaver et al. 2022, Am J Hum Genet (ClinGen SVI PP3/BP4)",
     "calibrated computational predictor thresholds"),
    ("PMC12208618", "Bergquist et al. 2025 (ClinGen SVI PP3/BP4, additional tools)",
     "calibration of further predictors"),
    ("PMC6938631", "Brnich et al. 2019, Genome Med (ClinGen SVI PS3/BS3)",
     "functional evidence"),
    ("PMC6188666", "Ghosh et al. 2018, Hum Mutat (ClinGen SVI BA1)", "BA1 and exceptions"),
    ("PMC10357475", "Walker et al. 2023, Am J Hum Genet (ClinGen SVI splicing)",
     "splicing evidence (PVS1, PP3, BP4, BP7, PS1)"),
    ("PMC6336098", "Tavtigian et al. 2018, Genet Med (Bayesian framework)",
     "Bayesian model of the guidelines"),
    ("PMC8011844", "Tavtigian et al. 2020, Hum Mutat (points system)", "points system"),
    ("PMC10806742", "Biesecker et al. 2024, Genet Med (ClinGen SVI PP1/BS4/PP4)",
     "segregation and phenotype specificity"),
    ("PMC6709533", "Biesecker & Harrison 2018, Genet Med (PP5/BP6)", "reputable source"),
    ("PMC7313390", "Riggs et al. 2020, Genet Med (ACMG/ClinGen CNV standards)",
     "copy-number variant interpretation"),
]
SKIP_SECTIONS = {"REF", "AUTH_CONT", "COMP_INT", "ACK_FUND", "SUPPL", "REVIEW_INFO"}


def _get(url: str, tries: int = 4, timeout: float = 90) -> httpx.Response:
    for i in range(tries):
        try:
            r = httpx.get(url, headers=UA, timeout=timeout, follow_redirects=True)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request,
                                            response=r)
            r.raise_for_status()
            return r
        except httpx.HTTPError:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))           # back off: NCBI rate-limits bursts
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- fetch
def fetch() -> int:
    (RAW / "pmc").mkdir(parents=True, exist_ok=True)
    (RAW / "cspec").mkdir(parents=True, exist_ok=True)
    failed = 0
    for pmc, cite, _ in PAPERS:
        try:
            r = _get(BIOC.format(pmc))
            # BioC returns a 200 with an "[Error] : No result..." body when a PMC article
            # is not in its open-access set; don't save that as if it were the paper.
            if not r.content.lstrip().startswith(b"["):
                print(f"  SKIP {pmc} {cite}: no open-access full text", file=sys.stderr)
                continue
            (RAW / "pmc" / f"{pmc}.json").write_bytes(r.content)
            print(f"  {pmc}  {cite}  ({len(r.content) // 1024} KB)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAILED {pmc} {cite}: {e}", file=sys.stderr)
        time.sleep(1.0)
    try:
        specs = _get(CSPEC_LIST).json()["data"]
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED ClinGen specification list: {e}", file=sys.stderr)
        return failed + 1
    released = [s for s in specs if s.get("status") in ("Released", "Approved For Release")]
    print(f"  ClinGen VCEP specifications: {len(released)} released of {len(specs)}")
    for s in released:
        sid = s["@id"].rsplit("/", 1)[1]
        try:
            r = _get(CSPEC_DOC.format(sid))
            (RAW / "cspec" / f"{sid}.json").write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAILED {sid}: {e}", file=sys.stderr)
        time.sleep(0.2)
    (RAW / "fetched.txt").write_text(date.today().isoformat())
    print(f"fetch done, {failed} failed")
    return failed


# --------------------------------------------------------------------------- build
def _clean(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"<sup>.*?</sup>", "", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)       # markdown links -> text
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("**", "").replace("_et al._", "et al.")
    return re.sub(r"[ \t]+", " ", t).strip()


def _chunks(text: str, size: int = 1400) -> list[str]:
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for para in re.split(r"(?<=[.;])\s+|\n+", text):
        if cur and len(cur) + len(para) > size:
            out.append(cur.strip())
            cur = ""
        cur += para + " "
    if cur.strip():
        out.append(cur.strip())
    return out


def _paper_passages(pmc: str, cite: str) -> list[tuple]:
    data = json.loads((RAW / "pmc" / f"{pmc}.json").read_text(encoding="utf-8"))
    coll = data[0] if isinstance(data, list) else data
    passages = coll["documents"][0]["passages"]
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"
    rows, section, buf = [], "", ""

    def flush():
        nonlocal buf
        for c in _chunks(buf.strip()):
            if len(c) > 60:
                rows.append((cite, section or "Text", "", c, url))
        buf = ""

    for p in passages:
        inf = p.get("infons", {})
        st, typ, text = inf.get("section_type", ""), inf.get("type", ""), _clean(p.get("text", ""))
        if st in SKIP_SECTIONS or not text:
            continue
        if typ.startswith("title"):
            flush()
            section = text[:120]
            continue
        if typ in ("table", "table_caption", "table_foot"):
            flush()
            label = inf.get("id") or "Table"
            for c in _chunks(text, 1800):
                rows.append((cite, f"{label} ({section})" if section else label, "", c, url))
            continue
        if len(buf) + len(text) > 1400:
            flush()
        buf += text + "\n"
    flush()
    return rows


def _cspec_passages(path: Path) -> tuple[list[tuple], list[tuple]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    d = d.get("data", d)
    sid = path.stem
    label = _clean(d.get("label", sid))
    url = f"https://cspec.genome.network/cspec/ui/svi/doc/{sid}"
    updated = (d.get("lastUpdated") or "")[:10]
    cite = f"ClinGen VCEP specification {sid}: {label} (updated {updated})"
    rows, vcep = [], []
    for rs in d.get("ruleSets") or []:
        genes = sorted({g.get("label", "").upper() for g in rs.get("genes") or [] if g.get("label")})
        gtxt = " ".join(genes)
        for g in genes:
            vcep.append((g, sid, label, d.get("version") or "", updated, url))
        for cc in rs.get("criteriaCodes") or []:
            code = cc.get("label", "")
            text = f"{code} ({', '.join(genes)}): {_clean(cc.get('description', ''))}"
            for s in cc.get("evidenceStrengths") or []:
                text += (f"\n- {s.get('label', '')}: {s.get('applicability', '')}"
                         + (f". {_clean(s.get('description', ''))}" if s.get("description") else ""))
            for c in _chunks(text, 1600):
                rows.append((cite, code, gtxt, c, url))
    return rows, vcep


def build() -> int:
    if not RAW.exists():
        print("nothing downloaded yet - run `fetch` first", file=sys.stderr)
        return 1
    tmp = INDEX.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("""
        CREATE VIRTUAL TABLE passages USING fts5(citation, section, genes, text, url UNINDEXED,
                                                 tokenize='unicode61');
        CREATE TABLE vcep (gene TEXT, spec_id TEXT, label TEXT, version TEXT, updated TEXT,
                           url TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);""")
    n_papers = 0
    for pmc, cite, _ in PAPERS:
        f = RAW / "pmc" / f"{pmc}.json"
        if not f.exists() or not f.read_bytes().lstrip().startswith(b"["):
            continue
        try:
            rows = _paper_passages(pmc, cite)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  SKIP {cite}: unreadable ({e})", file=sys.stderr)
            continue
        db.executemany("INSERT INTO passages VALUES (?,?,?,?,?)", rows)
        n_papers += 1
        print(f"  {cite}: {len(rows)} passages")
    n_specs = 0
    for f in sorted((RAW / "cspec").glob("*.json")):
        rows, vcep = _cspec_passages(f)
        db.executemany("INSERT INTO passages VALUES (?,?,?,?,?)", rows)
        db.executemany("INSERT INTO vcep VALUES (?,?,?,?,?,?)", vcep)
        n_specs += 1
    total = db.execute("SELECT count(*) FROM passages").fetchone()[0]
    fetched = (RAW / "fetched.txt").read_text() if (RAW / "fetched.txt").exists() else "?"
    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("fetched", fetched), ("papers", str(n_papers)), ("vcep_specs", str(n_specs)),
        ("passages", str(total))])
    db.execute("CREATE INDEX ix_vcep_gene ON vcep(gene)")
    db.commit()
    db.close()
    tmp.replace(INDEX)
    print(f"index built: {n_papers} papers, {n_specs} VCEP specifications, {total} passages "
          f"-> {INDEX}")
    return 0


# --------------------------------------------------------------------------- search
def available() -> bool:
    return INDEX.exists()


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{INDEX.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _fts_query(q: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) > 1]
    stop = {"the", "and", "for", "what", "does", "is", "of", "in", "to", "a", "an", "how",
            "when", "can", "be", "with", "on", "are", "this", "that", "do", "it", "or"}
    toks = [t for t in toks if t.lower() not in stop][:16]
    return " OR ".join(f'"{t}"' for t in toks)


def vcep_for_gene(gene: str) -> list[dict]:
    if not available() or not gene:
        return []
    with _con() as con:
        return [dict(r) for r in con.execute(
            "SELECT spec_id, label, version, updated, url FROM vcep WHERE gene = ?",
            (gene.upper(),))]


def search(query: str, gene: str | None = None, k: int = 4) -> dict:
    if not available():
        return {"error": "guideline index not built - run `python -m webui.knowledge.guidelines all`"}
    fq = _fts_query(query or "")
    if not fq:
        return {"error": "empty query"}
    hits, seen = [], set()
    with _con() as con:
        meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}
        groups = []
        if gene:
            # gene-specific expert-panel rules first: they override the general guidance
            groups.append((f'genes:"{gene.upper()}" AND ({fq})', 4))
        groups.append((fq, k))
        for match, lim in groups:
            try:
                rows = con.execute(
                    "SELECT citation, section, genes, text, url, bm25(passages, 2.0, 4.0, 1.0, 1.0) s "
                    "FROM passages WHERE passages MATCH ? ORDER BY s LIMIT ?",
                    (match, lim * 2)).fetchall()
            except sqlite3.OperationalError as e:
                return {"error": f"search failed: {e}"}
            for r in rows:
                key = (r["citation"], r["section"], r["text"][:80])
                if key in seen:
                    continue
                seen.add(key)
                hits.append({"citation": r["citation"], "section": r["section"],
                             "text": r["text"][:700], "url": r["url"]})
                if len([h for h in hits]) >= k + (4 if gene else 0):
                    break
    out = {"query": query, "passages": hits[: k + (3 if gene else 0)],
           "source": f"original guideline texts downloaded {meta.get('fetched', '?')}: "
                     f"{meta.get('papers')} papers + {meta.get('vcep_specs')} ClinGen VCEP "
                     f"specifications"}
    if gene:
        out["vcep_specifications_for_gene"] = vcep_for_gene(gene) or "none published"
    if not hits:
        out["note"] = "No passage matched. Try other wording, or the criterion code (e.g. PM2)."
    return out


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd in ("fetch", "all"):
        failed = fetch()
        if cmd == "fetch":
            return 1 if failed else 0
    if cmd in ("build", "all"):
        return build()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
