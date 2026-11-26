"""Genetics lookups for the chat: local database first, internet only where we must.

Local (llm/data/genetics.sqlite, built by build_db.py from the AI-Analyser resources):
    ClinVar classifications, gnomAD v4.1 gene constraint, ClinGen dosage sensitivity, HPO.
Online (only when GENETICS_ONLINE is on):
    NCBI E-utilities  - to turn a variant *name* (BRCA1:c.68_69del) into a ClinVar ID;
                        the classification itself is then read from the local copy.
    gnomAD GraphQL    - per-variant population frequencies (not stored locally).

Only gene symbols, variant names and HPO terms ever leave the machine - never anything the
user typed around them. Every result carries its source and release date so the model can
cite it, and every lookup failure is returned as data so the model can say "not found"
instead of guessing.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import httpx

from ..config import settings

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GNOMAD = "https://gnomad.broadinstitute.org/api"
UA = {"User-Agent": "advat-llm-chat/1.0"}

POP_NAMES = {
    "afr": "African/African American", "amr": "Admixed American", "asj": "Ashkenazi Jewish",
    "eas": "East Asian", "fin": "Finnish", "mid": "Middle Eastern", "nfe": "Non-Finnish European",
    "sas": "South Asian", "ami": "Amish", "remaining": "Remaining",
}


class LookupError_(RuntimeError):
    """A lookup failed in a way worth telling the model about."""


# --------------------------------------------------------------------------- database
def db_path() -> Path:
    return settings.genetics_db


def available() -> tuple[bool, str]:
    p = db_path()
    if not p.exists():
        return False, (f"genetics database not built ({p}). Build it with "
                       f"llm/webui/knowledge/build_db.py - see webui/README.md.")
    return True, "ready"


def _connect() -> sqlite3.Connection:
    ok, reason = available()
    if not ok:
        raise LookupError_(reason)
    con = sqlite3.connect(f"file:{db_path().as_posix()}?mode=ro", uri=True,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


_META: dict[str, str] | None = None


def meta() -> dict[str, str]:
    global _META
    if _META is None:
        with _connect() as con:
            _META = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}
    return _META


def _src_clinvar() -> str:
    return (f"ClinVar (public NCBI database, downloaded copy of the "
            f"{meta().get('clinvar_date', '?')} release, GRCh38)")


# --------------------------------------------------------------------------- online helpers
_NCBI_LOCK = threading.Lock()   # NCBI allows ~3 requests/s without an API key
_last_ncbi = 0.0


def _ncbi(endpoint: str, params: dict) -> dict:
    global _last_ncbi
    if not settings.genetics_online:
        raise LookupError_("online lookups are switched off (GENETICS_ONLINE=false)")
    with _NCBI_LOCK:
        wait = 0.35 - (time.time() - _last_ncbi)
        if wait > 0:
            time.sleep(wait)
        _last_ncbi = time.time()
    try:
        r = httpx.get(f"{NCBI}/{endpoint}", params={**params, "retmode": "json",
                                                    "tool": "advat-llm-chat"},
                      headers=UA, timeout=20)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise LookupError_(f"NCBI did not respond: {e}") from e


def _ncbi_search(term: str, retmax: int = 6) -> list[str]:
    return _ncbi("esearch.fcgi", {"db": "clinvar", "term": term,
                                  "retmax": retmax})["esearchresult"].get("idlist", [])


def _ncbi_summaries(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    res = _ncbi("esummary.fcgi", {"db": "clinvar", "id": ",".join(ids)})["result"]
    return {uid: res[uid] for uid in res.get("uids", [])}


# --------------------------------------------------------------------------- query parsing
_RS = re.compile(r"^\s*rs(\d+)\s*$", re.I)
_VCV = re.compile(r"^\s*(?:VCV0*|clinvar[:\s#]*(?:id)?[:\s#]*)(\d+)(?:\.\d+)?\s*$", re.I)
_COORD = re.compile(r"^\s*(?:chr)?([0-9]{1,2}|X|Y|MT?)[\s:\-_]+(\d+)[\s:\-_]+([ACGTN]+)"
                    r"[\s:\-_>/]+([ACGTN]+)\s*$", re.I)
_CHANGE = re.compile(r"\b([cgnmp]\.[^\s,;()]+(?:\([^)]*\))?)")
_ACC = re.compile(r"\b((?:NM|NR|NC|XM|ENST)_?\d+(?:\.\d+)?)", re.I)
_GENE = re.compile(r"\b([A-Z][A-Z0-9-]{1,15})\b")


def _parse_hgvs(q: str) -> tuple[str | None, str | None, str | None]:
    """-> (accession, gene, change) from things like 'NM_007294.4(BRCA1):c.68_69del',
    'BRCA1:c.68_69del', 'BRCA1 c.68_69del', 'CFTR p.Phe508del'."""
    m = _CHANGE.search(q)
    change = m.group(1).rstrip(".") if m else None
    acc = (_ACC.search(q) or [None, None])[1]
    gene = None
    paren = re.search(r"\(([A-Za-z0-9-]+)\)", q)
    if paren and not paren.group(1).lower().startswith(("p", "c")):
        gene = paren.group(1).upper()
    if not gene:
        head = q[: m.start()] if m else q
        for g in _GENE.findall(head.upper()):
            if not re.match(r"^(NM|NR|NC|XM|ENST)_?\d", g) and g not in {"HGVS", "VARIANT"}:
                gene = g
                break
    return acc, gene, change


# --------------------------------------------------------------------------- ClinVar
def _clinvar_rows(con, where: str, args: tuple) -> list[dict]:
    rows = con.execute(f"SELECT * FROM clinvar WHERE {where} LIMIT 10", args).fetchall()
    out = []
    for r in rows:
        out.append({
            "clinvar_variation_id": r["vid"],
            "clinvar_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{r['vid']}/",
            "position_grch38": f"{r['chrom']}-{r['pos']}-{r['ref']}-{r['alt']}",
            "hgvs_genomic": r["hgvsg"], "rsid": f"rs{r['rs']}" if r["rs"] else None,
            "genes": r["genes"].split("|") if r["genes"] else [],
            "classification": r["clnsig"], "review_status": r["revstat"],
            "review_stars": _stars(r["revstat"]),
            "conditions": [c for c in (r["conditions"] or "").split("|")
                           if c and c not in ("not provided", "not specified")][:5],
            "molecular_consequence": r["consequence"], "variant_type": r["vartype"],
        })
    return out


def _stars(revstat: str | None) -> int | None:
    if not revstat:
        return None
    s = revstat.lower()
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "multiple submitters" in s and "no conflicts" in s:
        return 2
    if "criteria provided" in s:
        return 1
    return 0


def _resolve_name(acc, gene, change) -> tuple[list[str], list[dict], str]:
    """Variant name -> ClinVar IDs via NCBI, strictest search first."""
    tries = []
    if acc and change:
        tries.append(f'"{acc}:{change}"')
    if gene and change:
        tries.append(f'"{gene}:{change}"')
        tries.append(f'{gene}[gene] AND "{change}"[varname]')
    for term in tries:
        ids = _ncbi_search(term)
        if ids:
            sums = _ncbi_summaries(ids[:5])
            cands = [{"clinvar_variation_id": int(uid), "clinvar_title": s.get("title"),
                      "online_classification": (s.get("germline_classification") or {})
                      .get("description")}
                     for uid, s in sums.items()]
            return ids[:5], cands, term
    return [], [], " | ".join(tries)


def _rs_fallback(con, rsid: str, out: dict) -> None:
    """The rsID is on no local ClinVar record (ClinVar files some variants under a merged
    rsID). Find where it sits, then match ClinVar by position or genomic HGVS."""
    # 1. gnomAD knows observed variants by rsID and returns the normalised position.
    gid = _gnomad_by_rsid(rsid)
    if gid:
        c, pos, ref, alt = gid.split("-")
        out["clinvar"] = _clinvar_rows(con, "chrom = ? AND pos = ? AND ref = ? AND alt = ?",
                                       (c, int(pos), ref, alt))
        out["position_grch38"] = gid
        out["matched_by"] = "rsID -> position via gnomAD (online), then local ClinVar"
        return
    # 2. Ensembl (slow). Its VCF positions are not always left-aligned the way ClinVar's
    #    and gnomAD's are, so match ClinVar on the genomic HGVS name instead.
    try:
        hits = _ensembl_recode(rsid)
    except LookupError_ as e:
        out["note"] = f"rsID not on any local ClinVar record, and {e}"
        return
    out["sources"].append("Ensembl variant recoder (online, position only)")
    out["matched_by"] = "rsID -> genomic HGVS via Ensembl (online), then local ClinVar"
    for vcf, hgvsg in hits:
        c, pos = vcf.split("-")[:2]
        rows = []
        if hgvsg:
            rows = _clinvar_rows(con, "chrom = ? AND pos BETWEEN ? AND ? AND hgvsg = ?",
                                 (c, int(pos) - 100, int(pos) + 100, hgvsg))
        out["clinvar"] += rows
    out["ensembl"] = [{"vcf": v, "hgvs_genomic": h} for v, h in hits]
    if not out["clinvar"] and hits:
        # No ClinVar record to borrow a normalised position from; Ensembl's may be
        # shifted for indels, so a gnomAD "not found" is not conclusive here.
        out["position_grch38"] = hits[0][0]
        out["gnomad_caveat"] = ("Position came from Ensembl and may not be left-aligned; for "
                                "an insertion/deletion a gnomAD 'not found' is not conclusive.")


def _gnomad_by_rsid(rsid: str) -> str | None:
    if not settings.genetics_online:
        return None
    try:
        r = httpx.post(GNOMAD, headers=UA, timeout=25, json={
            "query": "query($r:String!){variant(rsid:$r, dataset:gnomad_r4){variant_id}}",
            "variables": {"r": rsid}})
        v = (r.json().get("data") or {}).get("variant")
        return v["variant_id"] if v else None
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


def _ensembl_recode(rsid: str) -> list[tuple[str, str | None]]:
    """rsID -> [(chrom-pos-ref-alt, NC_ genomic HGVS)] via Ensembl (slow: often 15-25 s)."""
    if not settings.genetics_online:
        raise LookupError_("online lookups are switched off (GENETICS_ONLINE=false)")
    try:
        r = httpx.get(f"https://rest.ensembl.org/variant_recoder/human/{rsid}",
                      params={"vcf_string": 1, "fields": "vcf_string,hgvsg"},
                      headers={**UA, "Accept": "application/json"}, timeout=45)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise LookupError_(f"Ensembl did not respond: {e}") from e
    out: list[tuple[str, str | None]] = []
    for entry in data if isinstance(data, list) else []:
        for allele in entry.values():
            if not isinstance(allele, dict):
                continue
            vcfs = [v for v in allele.get("vcf_string") or []
                    if re.match(r"^[0-9XYM]+-\d+-[ACGTN]+-[ACGTN]+$", v)]
            hg = next((h for h in allele.get("hgvsg") or [] if h.startswith("NC_")), None)
            for v in vcfs:
                if v not in [x[0] for x in out]:
                    out.append((v, hg))
    return out[:4]


def _gnomad(variant_id: str) -> dict:
    """Population frequencies for one GRCh38 variant (chrom-pos-ref-alt) from gnomAD v4."""
    if not settings.genetics_online:
        return {"error": "online lookups are switched off (GENETICS_ONLINE=false)"}
    q = """query($id:String!){variant(variantId:$id, dataset:gnomad_r4){variant_id rsids
      exome{ac an homozygote_count populations{id ac an homozygote_count} faf95{popmax popmax_population}}
      genome{ac an homozygote_count populations{id ac an homozygote_count} faf95{popmax popmax_population}}}}"""
    try:
        # Short timeout: the local VEP-cache frequencies are the fallback.
        r = httpx.post(GNOMAD, json={"query": q, "variables": {"id": variant_id}},
                       headers=UA, timeout=8)
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"gnomAD did not respond: {e}"}
    v = (data.get("data") or {}).get("variant")
    if not v:
        msg = "; ".join(e.get("message", "") for e in data.get("errors", []))
        if "not found" in msg.lower():
            return {"found": False, "variant": variant_id,
                    "note": "Not observed in gnomAD v4.1 exomes or genomes. Absence can also "
                            "mean the region is poorly covered or hard to map."}
        return {"error": f"gnomAD: {msg or 'no data'}"}

    def total(p):
        if not p:
            return None
        faf = p.get("faf95") or {}
        return {"ac": p["ac"], "an": p["an"],
                "af": float(f"{p['ac'] / p['an']:.3g}") if p["an"] else None,
                "homozygotes": p["homozygote_count"],
                "faf95_popmax": float(f"{faf['popmax']:.3g}") if faf.get("popmax") else None,
                "faf95_popmax_population": POP_NAMES.get(faf.get("popmax_population") or "",
                                                         faf.get("popmax_population"))}

    # Exomes + genomes merged per population (as the gnomAD browser's total), only where
    # the variant is seen, as compact text: "0.0025 (228/91,054, 1 hom)". Populations not
    # listed have AC 0. Keeps the model's token use down.
    merged: dict[str, list[int]] = {}
    for p in (v.get("exome"), v.get("genome")):
        for x in (p or {}).get("populations") or []:
            # main gnomAD groups only: skip XX/XY splits and the small HGDP/1000G
            # sub-cohorts (ids like "hgdp:french"), whose tiny counts mislead
            if x["id"] not in POP_NAMES:
                continue
            m = merged.setdefault(x["id"], [0, 0, 0])
            m[0] += x["ac"]; m[1] += x["an"]; m[2] += x["homozygote_count"]  # noqa: E702
    pops = {POP_NAMES.get(k, k): f"{ac / an:.3g} ({ac}/{an:,}" + (f", {h} hom" if h else "")
            + ")" for k, (ac, an, h) in sorted(merged.items(),
                                               key=lambda kv: -(kv[1][0] / (kv[1][1] or 1)))
            if ac and an}
    ex, ge = total(v.get("exome")), total(v.get("genome"))
    ac = sum(x["ac"] for x in (ex, ge) if x)
    an = sum(x["an"] for x in (ex, ge) if x)
    return {"found": True, "variant": v["variant_id"],
            "combined_allele_frequency": float(f"{ac / an:.3g}") if an else None,
            "combined_allele_count": ac, "combined_allele_number": an,
            "exomes": ex, "genomes": ge,
            "populations_with_variant": pops,
            "url": f"https://gnomad.broadinstitute.org/variant/{v['variant_id']}?dataset=gnomad_r4"}


def lookup_variant(query: str) -> dict:
    """ClinVar (local) + gnomAD (online) for one variant, given as an HGVS name, rsID,
    ClinVar Variation ID/VCV, or GRCh38 chrom-pos-ref-alt."""
    query = (query or "").strip()
    if not query:
        return {"error": "empty query"}
    out: dict = {"query": query, "sources": [_src_clinvar()]}
    with _connect() as con:
        if m := _RS.match(query):
            out["matched_by"] = "rsID, local ClinVar"
            out["clinvar"] = _clinvar_rows(con, "rs = ?", (int(m.group(1)),))
            if not out["clinvar"]:
                _rs_fallback(con, f"rs{m.group(1)}", out)
        elif m := _VCV.match(query):
            out["matched_by"] = "ClinVar Variation ID, local ClinVar"
            out["clinvar"] = _clinvar_rows(con, "vid = ?", (int(m.group(1)),))
        elif m := _COORD.match(query):
            chrom = m.group(1).upper()
            chrom = "MT" if chrom == "M" else chrom
            pos, ref, alt = int(m.group(2)), m.group(3).upper(), m.group(4).upper()
            out["matched_by"] = "GRCh38 position, local ClinVar"
            out["clinvar"] = _clinvar_rows(con, "chrom = ? AND pos = ? AND ref = ? AND alt = ?",
                                           (chrom, pos, ref, alt))
            out["position_grch38"] = f"{chrom}-{pos}-{ref}-{alt}"
        else:
            acc, gene, change = _parse_hgvs(query)
            if not change:
                return {**out, "error": "could not read a variant from this. Give an HGVS name "
                        "with a gene or transcript (BRCA1:c.68_69del, NM_007294.4:c.68_69del), "
                        "an rsID, a ClinVar ID, or GRCh38 chrom-pos-ref-alt."}
            try:
                ids, cands, term = _resolve_name(acc, gene, change)
            except LookupError_ as e:
                return {**out, "error": f"{e}. Finding a variant by its HGVS name needs the "
                        "online NCBI search; offline, give the GRCh38 position "
                        "(chrom-pos-ref-alt), an rsID or a ClinVar ID instead."}
            out["matched_by"] = f"name search at NCBI ClinVar ({term}), facts from local ClinVar"
            out["name_matches"] = cands
            out["clinvar"] = []
            for uid in ids:
                out["clinvar"] += _clinvar_rows(con, "vid = ?", (int(uid),))
            local_ids = {r["clinvar_variation_id"] for r in out["clinvar"]}
            newer = [c for c in cands if c["clinvar_variation_id"] not in local_ids]
            # The online classification only matters for IDs missing from the local copy.
            for c in cands:
                if c["clinvar_variation_id"] in local_ids:
                    c.pop("online_classification", None)
            if newer:
                out["note_online_only"] = ("These ClinVar IDs exist online but are not in the "
                                           "local release; their classification shown is the "
                                           "current online one.")
            if not ids:
                out["note"] = ("No ClinVar record matched this name. It may be absent from "
                               "ClinVar, or named on a different transcript; try the RefSeq "
                               "transcript (NM_...) or the GRCh38 position.")

    # Best-reviewed record first (e.g. CFTR p.Phe508del has a 4-star practice-guideline
    # record and a rarer 1-star deletion with the same protein change).
    out["clinvar"] = sorted(out.get("clinvar") or [],
                            key=lambda r: -(r["review_stars"] or 0))
    if not out["clinvar"] and "error" not in out and "note" not in out:
        out["note"] = "Not in ClinVar (local release)."

    # Positions to annotate: ClinVar's are already left-aligned; a typed or Ensembl one
    # may not be, and VEP's normalisation step fixes that (and catches a wrong REF).
    positions = []
    for r in out.get("clinvar", []):
        if r["position_grch38"] not in positions:
            positions.append(r["position_grch38"])
    if not positions and out.get("position_grch38"):
        positions.append(out["position_grch38"])
    positions = positions[:2]

    if positions and settings.genetics_vep:
        positions = _annotate(out, positions)
    if positions and "error" not in out:
        _gnomad_online(out, positions)
        out["acmg_notes"] = _acmg_notes(out)
    if out.get("name_matches"):
        out["sources"].append("NCBI ClinVar search (online, used only to identify the variant)")
    return out


# --------------------------------------------------------------------------- local VEP
def _wsl_path(p: Path) -> str:
    """C:\\Users\\x\\f.py -> /mnt/c/Users/x/f.py"""
    p = p.resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def _vep(variants: list[dict]) -> dict:
    """Run vep_annotate.py inside WSL on [{'id','chrom','pos','ref','alt'}]."""
    script = _wsl_path(Path(__file__).with_name("vep_annotate.py"))
    try:
        p = subprocess.run(
            ["wsl", "-d", settings.wsl_distro, "-u", "root", "--", "python3", script,
             "--resources", settings.wsl_resources, "--tools-bin", settings.wsl_tools_bin],
            input=json.dumps({"variants": variants}), capture_output=True, text=True,
            timeout=150, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": f"local VEP did not run: {e}"}
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"error": f"local VEP failed: {(p.stderr or p.stdout)[-300:]}"}


_PLP = ("Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic")


def _split(pos: str) -> tuple[str, int, str, str]:
    c, p, r, a = pos.split("-")
    return c, int(p), r, a


def _annotate(out: dict, positions: list[str]) -> list[str]:
    """VEP the variant(s) plus pathogenic ClinVar variants nearby; returns the positions,
    left-aligned. Adds out['annotation'], out['clinvar_nearby'], out['capture_kit']."""
    chrom, pos, _r, _a = _split(positions[0])
    own = {r["clinvar_variation_id"] for r in out.get("clinvar", [])}
    with _connect() as con:
        near = con.execute(
            "SELECT vid, chrom, pos, ref, alt, clnsig, revstat FROM clinvar WHERE chrom = ? "
            "AND pos BETWEEN ? AND ? AND clnsig IN (?,?,?) LIMIT 25",
            (chrom, pos - 15, pos + 15, *_PLP)).fetchall()
    near = [n for n in near if n["vid"] not in own][:15]

    batch = [dict(zip(("chrom", "pos", "ref", "alt"), _split(p)), id=f"q{i}")
             for i, p in enumerate(positions)]
    batch += [{"id": f"n{i}", "chrom": n["chrom"], "pos": n["pos"], "ref": n["ref"],
               "alt": n["alt"]} for i, n in enumerate(near)]
    res = _vep(batch)
    if "results" not in res:
        out["annotation_error"] = res.get("error", "local VEP gave no result")
        return positions
    r = res["results"]

    ann, fixed = [], []
    for i, p in enumerate(positions):
        a = r.get(f"q{i}") or {"error": "no annotation"}
        if "error" in a and i == 0 and not out.get("clinvar"):
            out["error"] = a["error"]           # e.g. REF does not match GRCh38
            return []
        if i > 0:
            a.pop("pubmed", None)              # PubMed only for the main variant
        ann.append(a)
        fixed.append(a.get("position_grch38", p))
    out["annotation"] = ann[0] if len(ann) == 1 else ann
    out["sources"].append("VEP 116 + AlphaMissense (public tools/data, run locally; no release date)")

    # A typed position that VEP left-aligned may now match a ClinVar record.
    if not out.get("clinvar") and fixed[0] != positions[0]:
        with _connect() as con:
            out["clinvar"] = _clinvar_rows(con, "chrom = ? AND pos = ? AND ref = ? AND alt = ?",
                                           _split(fixed[0]))
        out["position_grch38"] = fixed[0]
        out["note_normalised"] = f"position left-aligned from {positions[0]} to {fixed[0]}"
        if out["clinvar"]:
            out.pop("note", None)

    # Pathogenic ClinVar variants at the same amino acid, split in code (not left to the
    # model) into the same protein change (PS1-type) and a different change at the same
    # residue (PM5-type), plus others nearby.
    main = ann[0]
    # PS1/PM5 need the comparison variant to be ESTABLISHED pathogenic, so only >=2-star
    # ClinVar records count; weaker ones are listed apart and labelled.
    same_change, other_change, weak, other = [], [], [], []
    is_missense = "missense" in (main.get("consequence") or "")
    for i, n in enumerate(near):
        a = r.get(f"n{i}") or {}
        stars = _stars(n["revstat"]) or 0
        label = (f"{a.get('hgvs_p') or a.get('hgvs_c') or n['chrom'] + '-' + str(n['pos'])} "
                 f"({n['clnsig']}, {stars}★, VCV {n['vid']})")
        same_gene = a.get("gene") == main.get("gene")
        same_residue = (main.get("protein_position") and same_gene
                        and a.get("protein_position") == main.get("protein_position"))
        # Same protein change matched on the p. name itself: an in-frame deletion can be
        # numbered from a different first residue (CFTR p.Phe508del: 507 vs 508).
        if same_gene and a.get("hgvs_p") and a.get("hgvs_p") == main.get("hgvs_p"):
            (same_change if stars >= 2 else weak).append("same protein change: " + label)
        elif same_residue and is_missense and "missense" in (a.get("consequence") or ""):
            (other_change if stars >= 2 else weak).append("other missense same residue: "
                                                          + label)
        else:
            other.append(label)
    out["clinvar_nearby"] = {
        "same_protein_change_established": [x.split(": ", 1)[1] for x in same_change],
        "different_missense_same_residue_established":
            [x.split(": ", 1)[1] for x in other_change] if is_missense else "n/a (not missense)",
        "same_residue_weak_records": weak,
        "other_within_15bp": other[:5] + ([f"+{len(other) - 5} more"] if len(other) > 5 else []),
        "note": "Pathogenic/likely pathogenic ClinVar variants (local) other than this one. "
                "'established' = 2 stars or more; 'weak' = 0-1 star, too weak a comparator "
                "for PS1/PM5."}

    c, p0 = _split(fixed[0])[:2]
    with _connect() as con:
        row = con.execute("SELECT start, end_ FROM capture WHERE chrom = ? AND end_ >= ? "
                          "ORDER BY end_ LIMIT 1", (c, p0)).fetchone()
        prev = con.execute("SELECT end_ FROM capture WHERE chrom = ? AND end_ < ? "
                           "ORDER BY end_ DESC LIMIT 1", (c, p0)).fetchone()
    kit = meta().get("capture_kit")
    if kit:
        inside = bool(row and row["start"] < p0 <= row["end_"])
        dist = 0 if inside else min([x for x in (
            (row["start"] + 1 - p0) if row else None,
            (p0 - prev["end_"]) if prev else None) if x is not None], default=None)
        out["capture_kit"] = {"kit": kit, "in_target": inside,
                              "distance_to_target_bp": dist}
    return fixed


_LOF = ("frameshift", "stop gained", "splice acceptor", "splice donor", "start lost")


def _acmg_notes(out: dict) -> list[str]:
    """Rule-level facts about which ACMG/AMP criteria the looked-up evidence bears on, and
    at what strength under ClinGen SVI guidance. Computed in code so the model does not
    have to remember the rules (it got PM2 strength and PS1/PM5 wrong from memory). These
    point at criteria; they do not apply them."""
    notes = []
    a = out.get("annotation")
    a = (a[0] if isinstance(a, list) else a) or {}
    cons = a.get("consequence") or ""

    # Frequency: PM2 / BA1 / BS1
    g = next((x for x in out.get("gnomad") or [] if x and "error" not in x), None)
    local = a.get("gnomad_local") or {}
    absent = (g and g.get("found") is False) or (not g and local and not (
        local.get("exomes_af") or local.get("genomes_af")))
    if absent:
        notes.append("Absent from gnomAD v4.1 -> PM2 may apply, at SUPPORTING strength "
                     "(PM2_Supporting) per ClinGen SVI 2020, not Moderate. Absence can also "
                     "reflect low coverage at this site.")
    else:
        afs = []
        if g and g.get("found"):
            afs = [g.get("combined_allele_frequency") or 0] + [
                (g.get(k) or {}).get("faf95_popmax") or 0 for k in ("exomes", "genomes")]
        elif local:
            afs = [local.get("exomes_af") or 0, local.get("genomes_af") or 0]
        top = max(afs, default=0)
        if top > 0.05:
            notes.append(f"Allele frequency {top:.3g} > 5% -> BA1 (stand-alone benign) "
                         "threshold met (ACMG 2015; check the ClinGen SVI BA1 exception list).")
        elif top:
            notes.append(f"Present in gnomAD (highest AF/FAF95 {top:.3g}) -> PM2 NOT MET. BS1 is "
                         "NOT ASSESSABLE from these data: it needs a disease-specific maximum "
                         "credible allele frequency (prevalence, inheritance, allelic "
                         "heterogeneity, penetrance) or a ClinGen VCEP threshold. Do not compute "
                         "or assume a threshold yourself unless the user supplies one.")
            cv0 = (out.get("clinvar") or [{}])[0]
            if (cv0.get("review_stars") or 0) >= 3 and "athogenic" in (
                    cv0.get("classification") or "") and "onflicting" not in (
                    cv0.get("classification") or ""):
                notes.append("An expert panel/practice guideline classifies this variant as "
                             "pathogenic at this frequency, so the frequency is evidently "
                             "compatible with the disease (e.g. a common recessive disorder). Do "
                             "NOT apply BS1 or BA1 against it.")

    # Variant type: PVS1 / PM4 / PP3-BP4
    if any(x in cons for x in _LOF):
        gene_bits = ""
        gene = a.get("gene")
        if gene:
            with _connect() as con:
                c = con.execute("SELECT loeuf FROM constraint_ WHERE gene = ?", (gene,)).fetchone()
                d = con.execute("SELECT hi_score FROM clingen_dosage WHERE gene = ?",
                                (gene,)).fetchone()
            gene_bits = (f" For {gene}: LOEUF {c['loeuf'] if c else 'n/a'}, ClinGen "
                         f"haploinsufficiency score {d['hi_score'] if d else 'not curated'}.")
        last = ""
        ex = a.get("exon") or ""
        if "/" in ex and ex.split("/")[0] == ex.split("/")[1]:
            last = " The variant is in the LAST exon, so it may escape NMD (lowers PVS1 strength)."
        notes.append("Predicted loss-of-function -> PVS1 candidate. Strength depends on LoF "
                     "being an established disease mechanism for the gene and on NMD/exon "
                     "position: apply the ClinGen SVI PVS1 decision tree (Abou Tayoun et al. "
                     "2018)." + gene_bits + last)
    elif "inframe" in cons:
        notes.append("In-frame length change -> PM4 candidate (not if in a repetitive region "
                     "without known function).")
    if a.get("alphamissense"):
        notes.append("Computational evidence here is ONE predictor (AlphaMissense). ClinGen SVI "
                     "(Pejaver et al. 2022) recommends a single calibrated predictor with "
                     "calibrated thresholds; PP3/BP4 strength follows the lab's calibration.")

    elif not any(x in cons for x in _LOF + ("synonymous",)):
        notes.append("No computational predictor in these data scores this variant type "
                     "(AlphaMissense covers missense only) -> PP3/BP4 NOT ASSESSABLE here "
                     "(not 'not met'); other calibrated tools may still apply.")

    # Same residue: PS1 / PM5 (established comparators only)
    nb = out.get("clinvar_nearby") or {}
    if nb.get("same_protein_change_established"):
        notes.append("An ESTABLISHED (>=2-star) P/LP ClinVar variant gives the SAME amino-acid "
                     "change by a different nucleotide change -> PS1-type evidence (PS1 is "
                     "defined for missense; check the nucleotide change does not affect "
                     "splicing differently).")
    if isinstance(nb.get("different_missense_same_residue_established"), list) and \
            nb["different_missense_same_residue_established"]:
        notes.append("An ESTABLISHED (>=2-star) P/LP missense differs at the same residue -> "
                     "PM5-type evidence (this is PM5, not PS1).")
    if nb.get("same_residue_weak_records"):
        notes.append("Same-residue ClinVar records with 0-1 star exist; they are too weak to "
                     "support PS1 or PM5 on their own.")

    cv = out.get("clinvar") or []
    if cv:
        notes.append("ClinVar's own classification is not an ACMG criterion (PP5/BP6 are not "
                     "recommended, Biesecker & Harrison 2018); use its underlying evidence.")
        top = cv[0]
        if (top.get("review_stars") or 0) >= 3:
            notes.append(f"ClinVar has an expert-panel/practice-guideline record "
                         f"({top['classification']}, {top['review_stars']} stars). Much of its "
                         "evidence (functional studies PS3/BS3, affected cases PS4/PM3, "
                         "segregation) lives in the literature, not in these databases, so the "
                         "criteria assessable here UNDERSTATE the real evidence. You MUST say "
                         "this in the answer, and must not present the database-only criteria "
                         "as the variant's overall picture.")
    notes.append("Not assessable from databases: de novo status (PS2/PM6), segregation "
                 "(PP1/BS4), functional studies (PS3/BS3), phase with another variant "
                 "(PM3/BP2), healthy homozygous/heterozygous adults (BS2), and PP4 unless the "
                 "user gave the patient's phenotype. Gene-specific ClinGen VCEP rules, where "
                 "they exist, override these general rules.")
    return notes


def _gnomad_online(out: dict, positions: list[str]) -> None:
    """Online gnomAD adds allele counts, homozygotes and FAF95. When it answers, drop the
    local VEP-cache frequencies for that variant so the model gets one set of numbers."""
    online = [_gnomad(p) for p in positions] if settings.genetics_online else []
    anns = out.get("annotation")
    anns = anns if isinstance(anns, list) else [anns] if anns else []
    used_local = False
    for i, p in enumerate(positions):
        g = online[i] if i < len(online) else None
        a = anns[i] if i < len(anns) else None
        if g and "error" not in g and a:
            a.pop("gnomad_local", None)
        elif a and a.get("gnomad_local"):
            used_local = True
    if online:
        out["gnomad"] = online
        out["sources"].append("gnomAD v4.1 (online API, live; no release date)")
    if used_local:
        out["sources"].append("gnomAD v4.1 frequencies (public, from the downloaded VEP cache)")


# --------------------------------------------------------------------------- genes
_BUCKETS = [("Pathogenic", "pathogenic"), ("Likely pathogenic", "likely_pathogenic"),
            ("Pathogenic/Likely pathogenic", "pathogenic_or_likely"),
            ("Uncertain significance", "uncertain"),
            ("Conflicting classifications of pathogenicity", "conflicting"),
            ("Likely benign", "likely_benign"), ("Benign", "benign"),
            ("Benign/Likely benign", "benign_or_likely")]


def lookup_gene(symbol: str) -> dict:
    """Constraint, dosage sensitivity, linked diseases and ClinVar counts for one gene."""
    gene = (symbol or "").strip().upper()
    if not gene:
        return {"error": "empty gene symbol"}
    m = meta()
    out: dict = {"gene": gene, "sources": []}
    with _connect() as con:
        c = con.execute("SELECT * FROM constraint_ WHERE gene = ?", (gene,)).fetchone()
        if c:
            out["gnomad_constraint"] = {
                "transcript": c["transcript"], "mane_select": bool(c["mane"]),
                # pLI rounded: the model misread 1.5e-34 as "pLI ~ 1"
                "pLI": round(c["pli"], 3) if c["pli"] is not None else None, "LOEUF": c["loeuf"], "lof_observed": c["lof_obs"],
                "lof_expected": c["lof_exp"], "lof_oe": c["lof_oe"],
                "missense_oe": c["mis_oe"], "missense_z": c["mis_z"],
                "synonymous_z": c["syn_z"], "flags": c["flags"] or None}
            out["sources"].append(m.get("gnomad_constraint", "gnomAD") + " gene constraint (public, downloaded copy)")
        d = con.execute("SELECT * FROM clingen_dosage WHERE gene = ?", (gene,)).fetchone()
        if d:
            out["clingen_dosage_sensitivity"] = {
                "haploinsufficiency_score": d["hi_score"],
                "haploinsufficiency": d["hi_desc"], "hi_disease": d["hi_disease"] or None,
                "triplosensitivity_score": d["ts_score"],
                "triplosensitivity": d["ts_desc"], "last_evaluated": d["last_evaluated"],
                "url": f"https://search.clinicalgenome.org/kb/gene-dosage?search={gene}"}
            out["sources"].append(f"ClinGen dosage sensitivity (public, downloaded {m.get('clingen_date')})")
        dis = con.execute(
            "SELECT g.disease_id, g.association, d.name, d.inheritance FROM gene_disease g "
            "LEFT JOIN disease d ON d.id = g.disease_id WHERE g.gene = ? LIMIT 40",
            (gene,)).fetchall()
        if dis:
            out["diseases"] = [{"id": r["disease_id"], "name": r["name"],
                                "inheritance": r["inheritance"] or None,
                                "association": r["association"]} for r in dis]
            out["sources"].append(f"HPO gene-disease annotations (public, release {m.get('hpo_version')})")
        counts = con.execute(
            "SELECT c.clnsig, COUNT(*) n FROM clinvar_gene g JOIN clinvar c ON c.rowid = g.rid "
            "WHERE g.gene = ? GROUP BY c.clnsig", (gene,)).fetchall()
        if counts:
            by = {r["clnsig"]: r["n"] for r in counts}
            summary = {key: by.get(label, 0) for label, key in _BUCKETS}
            summary["total_records"] = sum(by.values())
            out["clinvar_counts"] = summary
            out["sources"].append(_src_clinvar())
    validity, created = _gene_validity(gene)
    if validity is not None:
        out["clingen_gene_validity"] = validity or (
            "no ClinGen gene-disease validity curation for this gene")
        out["sources"].append(f"ClinGen gene-disease validity (public, file of {created})")
    if not out["sources"]:
        return {"gene": gene, "error": "gene symbol not found in any local source. Check the "
                "official HGNC symbol (old aliases such as GBA for GBA1 are not resolved)."}
    out["note"] = ("clingen_gene_validity = ClinGen's expert curation of how strong each "
                   "gene-disease link is (Definitive/Strong/Moderate/Limited/Disputed/Refuted). "
                   "'diseases' are HPO/OMIM/Orphanet annotations, which are not validity "
                   "assessments. clingen_dosage_sensitivity is a separate curation.")
    return out


_VALIDITY: tuple[dict, str] | None = None


def _gene_validity(gene: str) -> tuple[list[dict] | None, str]:
    """ClinGen gene-disease validity rows for a gene, from the downloaded CSV
    (webui/knowledge/fetch_public.py). (None, '') if the file has not been downloaded."""
    global _VALIDITY
    if _VALIDITY is None:
        path = settings.genetics_db.parent / "clingen_gene_validity.csv"
        if not path.exists():
            return None, ""
        import csv
        by_gene: dict[str, list[dict]] = {}
        created = ""
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                if row[0].startswith("FILE CREATED"):
                    created = row[0].split(":", 1)[1].strip()
                if len(row) < 10 or row[0] in ("GENE SYMBOL",) or row[0].startswith("+"):
                    continue
                by_gene.setdefault(row[0].upper(), []).append({
                    "disease": row[2], "mondo": row[3], "inheritance": row[4],
                    "classification": row[6], "date": row[8][:10], "gcep": row[9],
                    "url": row[7]})
        _VALIDITY = (by_gene, created)
    by_gene, created = _VALIDITY
    return by_gene.get(gene, []), created


# --------------------------------------------------------------------------- phenotypes
def lookup_phenotype(query: str) -> dict:
    """HPO term by ID or name: definition, synonyms, and genes annotated to it."""
    q = (query or "").strip()
    if not q:
        return {"error": "empty query"}
    m = meta()
    with _connect() as con:
        terms = []
        hp = re.match(r"^\s*HP[:_ ]?(\d{7})\s*$", q, re.I)
        if hp:
            tid = f"HP:{hp.group(1)}"
            alt = con.execute("SELECT id FROM hpo_alt WHERE alt_id = ?", (tid,)).fetchone()
            tid = alt["id"] if alt else tid
            terms = con.execute("SELECT * FROM hpo_term WHERE id = ?", (tid,)).fetchall()
        else:
            terms = con.execute("SELECT * FROM hpo_term WHERE name = ? COLLATE NOCASE",
                                (q,)).fetchall()
            if not terms:
                terms = con.execute(
                    "SELECT * FROM hpo_term WHERE name LIKE ? OR synonyms LIKE ? "
                    "ORDER BY length(name) LIMIT 6", (f"%{q}%", f"%{q}%")).fetchall()
        if not terms:
            return {"query": q, "error": "no HPO term matched this. Try a simpler or "
                    "alternative wording, or the HP: ID."}
        best = terms[0]
        genes = con.execute(
            "SELECT gene, COUNT(DISTINCT disease_id) n FROM gene_pheno WHERE hpo_id = ? "
            "GROUP BY gene ORDER BY n DESC, gene LIMIT 30", (best["id"],)).fetchall()
        n_genes = con.execute("SELECT COUNT(DISTINCT gene) FROM gene_pheno WHERE hpo_id = ?",
                              (best["id"],)).fetchone()[0]
        parents = [con.execute("SELECT id, name FROM hpo_term WHERE id = ?", (p,)).fetchone()
                   for p in (best["parents"] or "").split("|") if p]
    return {
        "query": q,
        "term": {"id": best["id"], "name": best["name"], "definition": best["definition"],
                 "synonyms": [s for s in (best["synonyms"] or "").split("|") if s][:10],
                 "parents": [f"{p['id']} {p['name']}" for p in parents if p],
                 "url": f"https://hpo.jax.org/browse/term/{best['id']}"},
        "other_matching_terms": [f"{t['id']} {t['name']}" for t in terms[1:]],
        "genes_annotated": {"total": n_genes,
                            "top_by_number_of_diseases": [f"{g['gene']} ({g['n']})"
                                                          for g in genes],
                            "note": "Direct annotations to this exact term only; genes "
                                    "annotated to more specific child terms are not counted."},
        "sources": [f"HPO (public, release {m.get('hpo_version')})"],
    }
