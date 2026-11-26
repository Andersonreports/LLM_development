"""Annotate a handful of variants with the AI-Analyser's local VEP setup. Runs INSIDE WSL
(where VEP, bcftools and the resources live); the chat calls it through `wsl.exe` with a
JSON request on stdin and reads a JSON answer from stdout. Standard library only.

    echo '{"variants":[{"id":"q","chrom":"17","pos":43124027,"ref":"ACT","alt":"A"}]}' | \
      python3 vep_annotate.py --resources /root/wes/resources \
                              --tools-bin /root/wes/tools/mamba/envs/wes/bin

Per variant it returns: the position after left-alignment against GRCh38 (bcftools norm,
which also catches a REF that does not match the genome), gene, MANE transcript, consequence,
HGVS c./p., exon/intron, AlphaMissense, gnomAD v4.1 frequencies from the VEP cache, dbSNP
rsIDs and PubMed IDs linked to the position.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
POPS = {"afr": "African", "amr": "Admixed American", "asj": "Ashkenazi Jewish",
        "eas": "East Asian", "fin": "Finnish", "mid": "Middle Eastern",
        "nfe": "Non-Finnish European", "sas": "South Asian", "ami": "Amish",
        "remaining": "Remaining"}


def run(cmd, env, stdin=None):
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env,
                          timeout=180)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", required=True, type=Path)
    ap.add_argument("--tools-bin", required=True, type=Path)
    a = ap.parse_args()
    req = json.load(sys.stdin)
    variants = req.get("variants") or []
    res, binp = a.resources, a.tools_bin
    fasta = res / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    env = dict(os.environ, PATH=f"{binp}{os.pathsep}{os.environ.get('PATH', '')}")
    out: dict[str, dict] = {}

    # 1. A tiny VCF; drop anything malformed before it reaches the tools.
    rows = []
    for v in variants:
        vid = str(v.get("id", ""))
        chrom = str(v.get("chrom", "")).upper().removeprefix("CHR")
        chrom = "MT" if chrom == "M" else chrom
        ref, alt = str(v.get("ref", "")).upper(), str(v.get("alt", "")).upper()
        try:
            pos = int(v.get("pos"))
        except (TypeError, ValueError):
            pos = 0
        if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", vid) or chrom not in CHROMS or pos < 1
                or not re.fullmatch(r"[ACGTN]{1,500}", ref)
                or not re.fullmatch(r"[ACGTN]{1,500}", alt)):
            out[vid or "?"] = {"error": "not a valid GRCh38 chrom-pos-ref-alt"}
            continue
        rows.append((chrom, pos, vid, ref, alt))
    if not rows:
        json.dump({"results": out}, sys.stdout)
        return 0

    with tempfile.TemporaryDirectory(prefix="chatvep_") as td:
        raw, norm = Path(td) / "in.vcf", Path(td) / "norm.vcf"
        with open(raw, "w") as fh:
            fh.write("##fileformat=VCFv4.2\n")
            fh.writelines(f"##contig=<ID={c}>\n" for c in CHROMS)
            fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for chrom, pos, vid, ref, alt in sorted(
                    rows, key=lambda r: (CHROMS.index(r[0]), r[1])):
                fh.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t.\n")

        # 2. Left-align against GRCh38 and check REF ("w" = warn, keep the record).
        p = run(["bcftools", "norm", "-f", str(fasta), "--check-ref", "w", "-o", str(norm),
                 str(raw)], env)
        if p.returncode != 0:
            json.dump({"error": f"bcftools norm failed: {p.stderr[-300:]}"}, sys.stdout)
            return 0
        mismatched = {}
        for line in p.stderr.splitlines():   # REF_MISMATCH  chrom  pos  given  genome
            f = line.split("\t")
            if f[0] == "REF_MISMATCH" and len(f) >= 5:
                mismatched[(f[1], f[2], f[3])] = f[4]
        keep = []
        for line in norm.read_text().splitlines():
            if line.startswith("#"):
                continue
            c, pos, vid, ref, alt = line.split("\t")[:5]
            orig = next((r for r in rows if r[2] == vid), None)
            key = (orig[0], str(orig[1]), orig[3]) if orig else None
            if key in mismatched:
                out[vid] = {"error": f"REF {orig[3]} does not match GRCh38 at "
                                     f"{orig[0]}:{orig[1]} (genome has {mismatched[key]}). "
                                     f"Check the position and genome build."}
                continue
            out[vid] = {"position_grch38": f"{c}-{pos}-{ref}-{alt}"}
            keep.append(line)
        if not keep:
            json.dump({"results": out}, sys.stdout)
            return 0
        norm.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                        + "\n".join(keep) + "\n")

        # 3. VEP, same cache/plugins as the AI-Analyser, one picked consequence per variant.
        plugins = sorted((binp.parent / "share").glob("ensembl-vep-*"))
        cmd = ["vep", "--offline", "--cache", "--dir_cache", str(res / "vep_cache"),
               "--assembly", "GRCh38", "--species", "homo_sapiens",
               "-i", str(norm), "-o", "STDOUT", "--json", "--no_stats", "--force_overwrite",
               "--symbol", "--numbers", "--canonical", "--mane", "--biotype", "--protein",
               "--variant_class", "--hgvs", "--fasta", str(fasta),
               "--af_gnomade", "--af_gnomadg", "--check_existing", "--pubmed",
               # flag (not drop) the other transcripts: AlphaMissense scores only the
               # UniProt transcript, which is often not the MANE one we report.
               "--flag_pick", "--pick_order",
               "mane_select,canonical,appris,tsl,biotype,ccds,rank,length"]
        am = res / "plugins" / "AlphaMissense_hg38.tsv.gz"
        if plugins and am.exists():
            cmd += ["--dir_plugins", str(plugins[-1]), "--plugin", f"AlphaMissense,file={am}"]
        p = run(cmd, env)
        if p.returncode != 0:
            json.dump({"results": out, "error": f"VEP failed: {p.stderr[-300:]}"}, sys.stdout)
            return 0
        for line in p.stdout.splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(rec.get("id"), {}).update(_compact(rec))
    json.dump({"results": out}, sys.stdout, separators=(",", ":"))
    return 0


def _compact(rec: dict) -> dict:
    tcs = rec.get("transcript_consequences") or [{}]
    tc = next((t for t in tcs if t.get("pick")), tcs[0])
    allele = tc.get("variant_allele")
    # AlphaMissense scores the genomic change; take it from whichever transcript has it.
    # (VEP's JSON nests plugin output: {"alphamissense": {"am_pathogenicity": ..}}.)
    am = lambda t: t.get("alphamissense") or (t if "am_pathogenicity" in t else None)  # noqa: E731
    am_tc = am(tc) or next((am(t) for t in tcs if am(t)), None)
    d = {"variant_class": rec.get("variant_class"),
         "consequence": rec.get("most_severe_consequence", "").replace("_", " ")}
    if tc:
        enst = tc.get("transcript_id")
        nm = tc.get("mane_select")
        hgvsc = tc.get("hgvsc")
        if hgvsc and nm and enst and hgvsc.startswith(enst):
            hgvsc = nm + hgvsc[hgvsc.index(":"):]          # MANE: same sequence, RefSeq name
        hgvsp = tc.get("hgvsp")
        d.update({
            "gene": tc.get("gene_symbol"), "transcript": nm or enst,
            "mane_select": bool(tc.get("mane")), "impact": tc.get("impact"),
            "consequence": ", ".join(tc.get("consequence_terms") or []).replace("_", " ")
            or d["consequence"],
            "hgvs_c": hgvsc,
            "hgvs_p": hgvsp.split(":", 1)[1].replace("%3D", "=") if hgvsp else None,
            "exon": tc.get("exon"), "intron": tc.get("intron"),
            "protein_position": tc.get("protein_start"),
            "amino_acids": tc.get("amino_acids"),
        })
        if am_tc:
            d["alphamissense"] = {"score": am_tc.get("am_pathogenicity"),
                                  "class": (am_tc.get("am_class") or "").replace("_", " ")}
    rs, pubmed, freq = [], [], {}
    for cv in rec.get("colocated_variants") or []:
        if str(cv.get("id", "")).startswith("rs"):
            rs.append(cv["id"])
            pubmed += cv.get("pubmed") or []
        f = (cv.get("frequencies") or {}).get(allele) if allele else None
        if f and not freq:
            freq = f
    d["rsids"] = rs[:3] or None
    if pubmed:
        ids = sorted(set(pubmed), reverse=True)            # higher PMID ~ more recent
        d["pubmed"] = {"count": len(ids), "most_recent": ids[:8],
                       "note": "PubMed IDs linked to this position's rsID in dbSNP/ClinVar; "
                               "they may concern another allele at the same position."}
    if freq:
        pops = {}
        for k, v in freq.items():
            m = re.fullmatch(r"gnomad([eg])_(\w+)", k)
            if m and v:
                pops.setdefault(POPS.get(m.group(2), m.group(2)), {})[
                    "exomes" if m.group(1) == "e" else "genomes"] = v
        top = sorted(((max(x.values()), name) for name, x in pops.items()), reverse=True)[:3]
        d["gnomad_local"] = {"exomes_af": freq.get("gnomade"), "genomes_af": freq.get("gnomadg"),
                             "highest_populations": [f"{n} {af:.3g}" for af, n in top]}
    else:
        d["gnomad_local"] = {"note": "no gnomAD v4.1 frequency in the VEP cache for this allele"}
    return {k: v for k, v in d.items() if v not in (None, "", [])}


if __name__ == "__main__":
    sys.exit(main())
