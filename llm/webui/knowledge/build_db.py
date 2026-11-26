"""Build one read-only SQLite database from the genetics resources the AI-Analyser already
downloaded, so the chat can look facts up locally instead of recalling them from memory.

Standard library only, so it runs with any Python 3.9+ (including WSL's system python3).
The resources live inside WSL (/root/wes/resources), which Windows cannot read directly,
so the usual way to run it is from Windows:

    wsl -d Ubuntu -u root -- python3 \
        /mnt/c/Users/Anderson/Desktop/Projects/LLM_development/llm/webui/knowledge/build_db.py \
        --resources /root/wes/resources \
        --out /mnt/c/Users/Anderson/Desktop/Projects/LLM_development/llm/data/genetics.sqlite

Inputs (paths relative to --resources):
    clinvar/clinvar.vcf.gz                         ClinVar GRCh38 VCF
    gene/gnomad.v4.1.constraint_metrics.tsv        gnomAD v4.1 gene constraint
    gene/ClinGen_gene_curation_list_GRCh38.tsv     ClinGen dosage sensitivity (HI/TS)
    hpo/hp.obo, hpo/phenotype.hpoa,
    hpo/genes_to_disease.txt, hpo/genes_to_phenotype.txt   HPO

The database is built in a temp dir and moved into place at the end, so a half-built file
never replaces a working one.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

-- one row per ClinVar VCF record (a Variation ID can map to one allele)
CREATE TABLE clinvar (
    vid INTEGER,           -- ClinVar Variation ID (VCF ID column)
    chrom TEXT, pos INTEGER, ref TEXT, alt TEXT,
    rs INTEGER,            -- dbSNP rs number without the 'rs'
    genes TEXT,            -- GENEINFO symbols joined with '|'
    clnsig TEXT,           -- germline classification (CLNSIG), underscores -> spaces
    revstat TEXT,          -- review status (CLNREVSTAT)
    conditions TEXT,       -- CLNDN, '|' separated
    consequence TEXT,      -- MC, molecular consequence names
    hgvsg TEXT,            -- CLNHGVS (genomic)
    vartype TEXT           -- CLNVC
);
CREATE TABLE clinvar_gene (gene TEXT, rid INTEGER);   -- rid = clinvar.rowid

CREATE TABLE constraint_ (
    gene TEXT PRIMARY KEY, transcript TEXT, mane INTEGER,
    pli REAL, lof_oe REAL, loeuf REAL, lof_obs REAL, lof_exp REAL,
    mis_oe REAL, mis_z REAL, syn_z REAL, flags TEXT
);

CREATE TABLE clingen_dosage (
    gene TEXT PRIMARY KEY, gene_id TEXT, cytoband TEXT, location TEXT,
    hi_score TEXT, hi_desc TEXT, hi_pmids TEXT, hi_disease TEXT,
    ts_score TEXT, ts_desc TEXT, ts_pmids TEXT, ts_disease TEXT,
    last_evaluated TEXT
);

CREATE TABLE hpo_term (id TEXT PRIMARY KEY, name TEXT, definition TEXT, synonyms TEXT,
                       parents TEXT);
CREATE TABLE hpo_alt (alt_id TEXT PRIMARY KEY, id TEXT);
CREATE TABLE disease (id TEXT PRIMARY KEY, name TEXT, inheritance TEXT);
CREATE TABLE gene_disease (gene TEXT, disease_id TEXT, association TEXT);
CREATE TABLE gene_pheno (gene TEXT, hpo_id TEXT, frequency TEXT, disease_id TEXT);

-- the lab's exome capture kit target regions (BED, 0-based start, end exclusive)
CREATE TABLE capture (chrom TEXT, start INTEGER, end_ INTEGER);
"""

INDEXES = """
CREATE INDEX ix_clinvar_vid ON clinvar(vid);
CREATE INDEX ix_clinvar_pos ON clinvar(chrom, pos);
CREATE INDEX ix_clinvar_rs ON clinvar(rs);
CREATE INDEX ix_clinvar_gene ON clinvar_gene(gene);
CREATE INDEX ix_hpo_name ON hpo_term(name COLLATE NOCASE);
CREATE INDEX ix_gd_gene ON gene_disease(gene);
CREATE INDEX ix_gp_gene ON gene_pheno(gene);
CREATE INDEX ix_gp_hpo ON gene_pheno(hpo_id);
CREATE INDEX ix_capture ON capture(chrom, end_);
"""


def load_capture(db: sqlite3.Connection, bed: Path) -> None:
    log(f"capture kit: reading {bed}")
    rows = []
    with open(bed, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")) or not line.strip():
                continue
            c = line.split("\t")
            chrom = c[0].removeprefix("chr")
            rows.append(("MT" if chrom == "M" else chrom, int(c[1]), int(c[2])))
    db.executemany("INSERT INTO capture VALUES (?,?,?)", rows)
    db.execute("INSERT INTO meta VALUES ('capture_kit', ?)", (bed.stem,))
    log(f"capture kit: {len(rows):,} target regions ({bed.stem})")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _f(x: str):
    """Float or None (gnomAD writes 'NA' / '' for missing)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- ClinVar
def load_clinvar(db: sqlite3.Connection, path: Path) -> None:
    log(f"ClinVar: reading {path}")
    file_date = ""
    rows, gene_rows, n = [], [], 0
    rid = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("##fileDate="):
                    file_date = line.strip().split("=", 1)[1]
                continue
            chrom, pos, vid, ref, alt, _q, _f2, info = line.rstrip("\n").split("\t")[:8]
            d = {}
            for kv in info.split(";"):
                k, _, v = kv.partition("=")
                if k in ("CLNSIG", "CLNREVSTAT", "CLNDN", "GENEINFO", "MC", "CLNHGVS",
                         "RS", "CLNVC"):
                    d[k] = v
            genes = [g.split(":")[0] for g in d.get("GENEINFO", "").split("|") if g]
            mc = "|".join(x.split("|")[-1] for x in d.get("MC", "").split(",") if x)
            rs = d.get("RS", "").split("|")[0]
            rid += 1
            rows.append((rid, int(vid) if vid.isdigit() else None, chrom, int(pos), ref, alt,
                         int(rs) if rs.isdigit() else None, "|".join(genes),
                         d.get("CLNSIG", "").replace("_", " ") or None,
                         d.get("CLNREVSTAT", "").replace("_", " ") or None,
                         d.get("CLNDN", "").replace("_", " ") or None,
                         mc.replace("_", " ") or None, d.get("CLNHGVS") or None,
                         d.get("CLNVC", "").replace("_", " ") or None))
            gene_rows.extend((g, rid) for g in genes)
            if len(rows) >= 200_000:
                _flush_clinvar(db, rows, gene_rows)
                n += len(rows)
                rows, gene_rows = [], []
                log(f"  {n:,} records")
    _flush_clinvar(db, rows, gene_rows)
    n += len(rows)
    db.execute("INSERT INTO meta VALUES ('clinvar_date', ?)", (file_date,))
    db.execute("INSERT INTO meta VALUES ('clinvar_records', ?)", (str(n),))
    log(f"ClinVar: {n:,} records, release {file_date}")


def _flush_clinvar(db, rows, gene_rows):
    db.executemany("INSERT INTO clinvar(rowid, vid, chrom, pos, ref, alt, rs, genes, clnsig, "
                   "revstat, conditions, consequence, hgvsg, vartype) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.executemany("INSERT INTO clinvar_gene VALUES (?,?)", gene_rows)


# --------------------------------------------------------------------------- gnomAD
def load_constraint(db: sqlite3.Connection, path: Path) -> None:
    log(f"gnomAD constraint: reading {path}")
    best: dict[str, tuple] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        h = {name: i for i, name in enumerate(header)}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            gene = c[h["gene"]]
            mane = c[h["mane_select"]] == "true"
            canonical = c[h["canonical"]] == "true"
            if not (mane or canonical):
                continue
            row = (gene, c[h["transcript"]], int(mane),
                   _f(c[h["lof.pLI"]]), _f(c[h["lof.oe"]]), _f(c[h["lof.oe_ci.upper"]]),
                   _f(c[h["lof.obs"]]), _f(c[h["lof.exp"]]),
                   _f(c[h["mis.oe"]]), _f(c[h["mis.z_score"]]), _f(c[h["syn.z_score"]]),
                   c[h["constraint_flags"]] if "constraint_flags" in h else "")
            # Prefer the MANE Select transcript; fall back to the canonical one.
            if gene not in best or (mane and not best[gene][2]):
                best[gene] = row
    db.executemany("INSERT INTO constraint_ VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", best.values())
    db.execute("INSERT INTO meta VALUES ('gnomad_constraint', 'gnomAD v4.1')")
    log(f"gnomAD constraint: {len(best):,} genes")


# --------------------------------------------------------------------------- ClinGen
def load_clingen(db: sqlite3.Connection, path: Path) -> None:
    log(f"ClinGen dosage: reading {path}")
    date, header, rows = "", None, []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if i == 1 and line.startswith("#"):
                date = line.lstrip("#").strip()
            if line.startswith("#Gene Symbol"):
                header = line.lstrip("#").split("\t")
                continue
            if line.startswith("#") or not line.strip() or header is None:
                continue
            c = dict(zip(header, line.split("\t")))

            def pmids(prefix):
                return ",".join(c.get(f"{prefix} PMID{i}", "") for i in range(1, 7)
                                if c.get(f"{prefix} PMID{i}"))
            rows.append((c.get("Gene Symbol"), c.get("Gene ID"), c.get("cytoBand"),
                         c.get("Genomic Location"),
                         c.get("Haploinsufficiency Score"), c.get("Haploinsufficiency Description"),
                         pmids("Haploinsufficiency"), c.get("Haploinsufficiency Disease ID"),
                         c.get("Triplosensitivity Score"), c.get("Triplosensitivity Description"),
                         pmids("Triplosensitivity"), c.get("Triplosensitivity Disease ID"),
                         c.get("Date Last Evaluated")))
    db.executemany("INSERT OR REPLACE INTO clingen_dosage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   rows)
    db.execute("INSERT INTO meta VALUES ('clingen_date', ?)", (date,))
    log(f"ClinGen dosage: {len(rows):,} genes ({date})")


# --------------------------------------------------------------------------- HPO
def load_hpo(db: sqlite3.Connection, hpo_dir: Path) -> None:
    log("HPO: reading hp.obo")
    terms, alts, cur = [], [], None
    version = ""

    def flush():
        if cur and cur.get("id", "").startswith("HP:") and not cur.get("obsolete"):
            terms.append((cur["id"], cur.get("name", ""), cur.get("def", ""),
                          "|".join(cur.get("syn", [])), "|".join(cur.get("is_a", []))))
            alts.extend((a, cur["id"]) for a in cur.get("alt", []))

    with open(hpo_dir / "hp.obo", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip()
            if line == "[Term]":
                flush()
                cur = {}
                continue
            if line.startswith("[") and line.endswith("]"):
                flush()
                cur = None
                continue
            if cur is None or ": " not in line:
                continue
            k, v = line.split(": ", 1)
            if k == "id":
                cur["id"] = v
            elif k == "name":
                cur["name"] = v
            elif k == "def":
                cur["def"] = v.split('"')[1] if '"' in v else v
            elif k == "synonym":
                cur.setdefault("syn", []).append(v.split('"')[1] if '"' in v else v)
            elif k == "is_a":
                cur.setdefault("is_a", []).append(v.split(" ")[0])
            elif k == "alt_id":
                cur.setdefault("alt", []).append(v)
            elif k == "is_obsolete" and v == "true":
                cur["obsolete"] = True
    flush()
    db.executemany("INSERT OR REPLACE INTO hpo_term VALUES (?,?,?,?,?)", terms)
    db.executemany("INSERT OR REPLACE INTO hpo_alt VALUES (?,?)", alts)
    names = {t[0]: t[1] for t in terms}
    log(f"HPO: {len(terms):,} terms")

    log("HPO: reading phenotype.hpoa (disease names + inheritance)")
    dname: dict[str, str] = {}
    inh: dict[str, set] = {}
    hpoa_version = ""
    with open(hpo_dir / "phenotype.hpoa", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#version:"):
                    hpoa_version = line.split(":", 1)[1].strip()
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 11 or c[0] == "database_id":
                continue
            dname.setdefault(c[0], c[1])
            if c[10] == "I":
                inh.setdefault(c[0], set()).add(names.get(c[3], c[3]))
    db.executemany("INSERT OR REPLACE INTO disease VALUES (?,?,?)",
                   [(d, n, "|".join(sorted(inh.get(d, ())))) for d, n in dname.items()])

    log("HPO: reading genes_to_disease.txt")
    rows = []
    with open(hpo_dir / "genes_to_disease.txt", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 4:
                rows.append((c[1], c[3], c[2]))
    db.executemany("INSERT INTO gene_disease VALUES (?,?,?)", rows)

    log("HPO: reading genes_to_phenotype.txt")
    rows = []
    with open(hpo_dir / "genes_to_phenotype.txt", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) >= 6:
                rows.append((c[1], c[2], c[4], c[5]))
    db.executemany("INSERT INTO gene_pheno VALUES (?,?,?,?)", rows)
    db.execute("INSERT INTO meta VALUES ('hpo_version', ?)", (hpoa_version or version,))
    log(f"HPO: {len(dname):,} diseases, {len(rows):,} gene-phenotype links")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resources", required=True, type=Path,
                    help="the AI-Analyser resources dir, e.g. /root/wes/resources")
    ap.add_argument("--out", required=True, type=Path, help="where to write genetics.sqlite")
    args = ap.parse_args()
    res = args.resources

    needed = [res / "clinvar/clinvar.vcf.gz", res / "gene/gnomad.v4.1.constraint_metrics.tsv",
              res / "gene/ClinGen_gene_curation_list_GRCh38.tsv", res / "hpo/hp.obo",
              res / "hpo/phenotype.hpoa", res / "hpo/genes_to_disease.txt",
              res / "hpo/genes_to_phenotype.txt"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print("missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    t0 = time.time()
    tmpdir = Path(tempfile.mkdtemp(prefix="genetics_db_"))
    tmp = tmpdir / "genetics.sqlite"
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    db.executescript(SCHEMA)
    load_clinvar(db, res / "clinvar/clinvar.vcf.gz")
    load_constraint(db, res / "gene/gnomad.v4.1.constraint_metrics.tsv")
    load_clingen(db, res / "gene/ClinGen_gene_curation_list_GRCh38.tsv")
    load_hpo(db, res / "hpo")
    beds = sorted((res / "targets").glob("*.bed"))
    if beds:
        load_capture(db, beds[0])
    db.execute("INSERT INTO meta VALUES ('built_at', ?)",
               (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),))
    db.execute("INSERT INTO meta VALUES ('assembly', 'GRCh38')")
    db.commit()
    log("building indexes")
    db.executescript(INDEXES)
    db.execute("ANALYZE")
    db.commit()
    db.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".partial")
    shutil.copyfile(tmp, partial)
    os.replace(partial, args.out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    log(f"done: {args.out} ({args.out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
