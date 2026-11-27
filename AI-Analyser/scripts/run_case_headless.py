"""Headless ADVAT runner: VCF + clinical history -> full pipeline on the LOCAL model ->
report + verdict. For blind proof runs (no web UI, reproducible).

    python run_case_headless.py --vcf X.vcf --clinical hist.txt --age 5y --sex M \
        --indication "..." --accession PROOF1 --out OUTDIR
"""
import argparse
import json
import sys
from pathlib import Path

from wes.db import Case, Sample, SessionLocal, init_db
from wes import ai, pipeline, report
from wes.vcf import read_header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--clinical", default="")   # path to a .txt, or literal text
    ap.add_argument("--age", default="")
    ap.add_argument("--sex", default="")
    ap.add_argument("--indication", default="")
    ap.add_argument("--accession", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vcf = Path(a.vcf).resolve()
    if not vcf.exists():
        print(f"VCF not found: {vcf}", file=sys.stderr)
        return 1
    clinical = a.clinical
    if clinical and Path(clinical).exists():
        clinical = Path(clinical).read_text(encoding="utf-8", errors="replace")

    print(f"backend: {ai.describe_backend()}", flush=True)
    print(f"AI available: {ai.available()}", flush=True)

    init_db()
    db = SessionLocal()
    sample_name = read_header(vcf).samples[0]
    acc = a.accession or ("PROOF_" + vcf.stem[:24])
    # start clean if this accession was run before
    old = db.query(Case).filter(Case.accession == acc).first()
    if old:
        db.delete(old)
        db.commit()
    case = Case(accession=acc, case_type="single", clinical_history=clinical,
                indication=a.indication or None, status="created")
    case.samples = [Sample(role="proband", name=acc, age=a.age or None, sex=a.sex or None,
                           affected=True, vcf_path=str(vcf), vcf_sample=sample_name)]
    db.add(case)
    db.commit()
    cid = case.id
    print(f"case id={cid} accession={acc} sample={sample_name}", flush=True)
    db.close()

    pipeline.run_case(cid)

    db = SessionLocal()
    case = db.get(Case, cid)
    res = case.results or {}
    air = case.ai_result or {}
    print("\n================ RESULT ================", flush=True)
    print(f"status        : {case.status}")
    print(f"error         : {case.error}")
    print(f"interpreted by: {ai.describe_backend()}")
    print(f"\nVERDICT")
    print(f"  rule-engine flag : {case.rule_flag}")
    print(f"  AI model flag    : {air.get('result_flag') or case.ai_flag}")
    print(f"  summary line     : {air.get('summary_line')}")
    print(f"  rationale        : {air.get('flag_rationale')}")

    cands = res.get("candidates") or []
    print(f"\nACMG TABLE (all {len(cands)} candidates)")
    for v in cands:
        a = v.get("acmg") or {}
        crit = "+".join(f"{c['code']}({c['strength']})" for c in (a.get("criteria") or []))
        print(f"  {v.get('gene'):<10} {v.get('hgvsp') or v.get('hgvsc') or '':<18} "
              f"{a.get('short') or '':<4} {a.get('classification') or '':<20} {crit}")

    findings = air.get("findings") or []
    if findings:
        print(f"\nAI-SELECTED FINDINGS ({len(findings)})")
        for f in findings:
            print(f"  {f.get('gene')} {f.get('variant') or ''} -> "
                  f"section={f.get('report_section')} class={f.get('classification')}")
    if air.get("clinical_interpretation"):
        print(f"\nCLINICAL INTERPRETATION (model):\n  {air['clinical_interpretation'][:800]}")
    if air.get("reviewer_attention"):
        print(f"\nFLAGGED FOR REVIEWER:\n  {air['reviewer_attention']}")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{acc}_result.json").write_text(
        json.dumps({"accession": acc, "rule_flag": case.rule_flag, "ai_flag": case.ai_flag,
                    "final_flag": case.final_flag, "results": res,
                    "ai_result": case.ai_result, "hpo_terms": case.hpo_terms,
                    "backend": ai.describe_backend()}, indent=1, default=str),
        encoding="utf-8")
    try:
        html = report.render_html(case, final=False)
        (outdir / f"{acc}_report.html").write_text(html, encoding="utf-8")
        print(f"\nwrote {acc}_report.html")
        try:
            from weasyprint import HTML
            HTML(string=html, base_url=str(outdir)).write_pdf(str(outdir / f"{acc}_report.pdf"))
            print(f"wrote {acc}_report.pdf")
        except Exception as e:  # noqa: BLE001
            print(f"PDF skipped ({type(e).__name__}: {e}); HTML is fine for review")
    except Exception as e:  # noqa: BLE001
        print(f"report render failed ({type(e).__name__}: {e}); result.json has the findings")
    db.close()
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
