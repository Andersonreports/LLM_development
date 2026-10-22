"""Builds the clinical report (HTML + PDF) from a reviewed case, in the laboratory's
standard "Whole Exome Sequencing Analysis" layout."""
from __future__ import annotations

import base64
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import report_text as T
from .acmg import SHORT
from .config import settings
from .db import Case
from .phenotype import read_coverage
from .pipeline import _all_variants
from .resources import ACMG_SF_V3_2

TEMPLATES = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))


def _date(v: str | None) -> str:
    """ISO date from the form -> dd/mm/yyyy (other text is left as typed)."""
    if not v:
        return ""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v.strip())
    return f"{m.group(3)}/{m.group(2)}/{m.group(1)}" if m else v


def _age(v: str | None) -> str:
    if not v:
        return ""
    v = v.strip()
    return f"{v} Years" if re.fullmatch(r"\d+(\.\d+)?", v) else v


_env.filters["date"] = _date
_env.filters["age"] = _age

SUBTITLES = {"single": "", "trio": "Trio analysis (proband and both parents)", "couple": "Couple carrier screening"}

RESULT_TEXT = {
    ("single", "POSITIVE"): "Pathogenic or likely pathogenic variant(s) causative of the reported phenotype was detected",
    ("single", "INCONCLUSIVE"): "Variant(s) of uncertain significance in gene(s) associated with the reported phenotype was detected",
    ("single", "NEGATIVE"): T.NEGATIVE_RESULT,
    ("couple", "POSITIVE"): "Increased reproductive risk: pathogenic or likely pathogenic variant(s) were detected in the same gene in both partners, or in an X-linked gene in the female partner",
    ("couple", "CARRIER"): "Carrier status was detected; no shared-gene reproductive risk was identified",
    ("couple", "NEGATIVE"): "No pathogenic or likely pathogenic carrier variant(s) were detected in either partner",
}

ROLE_LABEL = {"proband": "Proband", "father": "Father", "mother": "Mother",
              "partner1": "Partner 1", "partner2": "Partner 2"}
ZYG_LABEL = {"het": "Heterozygous", "hom": "Homozygous", "hemizygous": "Hemizygous", "ref": "Not detected",
             "no_call": "No call", "homoplasmic": "Homoplasmic", "heteroplasmic": "Heteroplasmic", None: "-"}
ORIGIN_LABEL = {"father": "Paternal", "mother": "Maternal", "both": "Biparental", "de_novo": "De novo",
                "unknown": "Not determined", None: None}


def _default_strength(c: dict) -> bool:
    defaults = {"PVS": "VeryStrong", "PS": "Strong", "PM": "Moderate", "PP": "Supporting",
                "BA": "StandAlone", "BS": "Strong", "BP": "Supporting"}
    prefix = "PVS" if c["code"].startswith("PVS") else c["code"][:2]
    return defaults.get(prefix) == c["strength"]


def _row(v: dict, f: dict, roles: list[str]) -> dict:
    zyg = {ROLE_LABEL[r]: ZYG_LABEL.get(v["zygosity"].get(r), v["zygosity"].get(r)) for r in roles if r in v["zygosity"]}
    short = SHORT.get(f.get("classification_override"), f["acmg"]["short"])
    return {
        "gene": v["gene"],
        "transcript": v.get("transcript") or "",
        "hgvsc": v.get("hgvsc") or f"{v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}",
        "hgvsp": v.get("hgvsp") or "",
        "genomic": f"chr{v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}",
        "rsid": v.get("rsid"),
        "consequence": v.get("consequence_str"),
        "exon": f"Exon {v['exon']}" if v.get("exon") else (f"Intron {v['intron']}" if v.get("intron") else ""),
        "zygosity": zyg,
        "origin": ORIGIN_LABEL.get(v.get("origin")),
        "pop_af": v.get("pop_af"),
        "disease": f.get("disease") or "",
        "disease_inheritance": f.get("disease_inheritance") or "",
        "classification": f.get("classification_override") or f["acmg"]["classification"],
        "short": short,
        "criteria": ", ".join(
            c["code"] + ("" if _default_strength(c) else f"_{c['strength']}") for c in f["acmg"]["criteria"]),
        "interpretation": f.get("interpretation") or "",
    }


def _vlabel(v: dict, fallback: str) -> str:
    c = v.get("hgvsc") or fallback
    return f"{c} ({v['hgvsp']})" if v.get("hgvsp") else c


def _img(path: Path) -> str | None:
    if not path or not Path(path).exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def _gene_table(case: Case, review: dict, sections: dict) -> dict:
    genes = list(review.get("gene_list") or (case.results or {}).get("gene_list") or [])
    for rows in sections.values():
        genes += [r["gene"] for r in rows]
    genes = sorted(set(g.upper() for g in genes))
    cov = read_coverage(case.coverage_path) if case.coverage_path else {}
    return {"genes": genes, "coverage": {g: cov.get(g) for g in genes} if cov else None}


def build_context(case: Case, final: bool) -> dict:
    review = case.review or {}
    pool = _all_variants(case.results or {})
    roles = [s.role for s in case.samples]
    sections: dict[str, list] = {"primary": [], "vus": [], "carrier": [], "secondary": []}
    for f in review.get("findings", []):
        if not f.get("include", True) or f.get("report_section") not in sections:
            continue
        v = pool.get(f["variant_key"])
        if v:
            sections[f["report_section"]].append(_row(v, f, roles))

    ctype = case.case_type
    flag = case.final_flag or review.get("flag") or case.ai_flag or case.rule_flag or "NEGATIVE"
    rkey = ("couple" if ctype == "couple" else "single", flag)

    at_risk = []
    if ctype == "couple":
        for r in (case.results or {}).get("couple", {}).get("at_risk", []):
            at_risk.append({
                "gene": r["gene"], "mode": r["mode"], "risk": r["risk"],
                "diseases": "; ".join(d["name"] for d in r["diseases"][:2]),
                "p1": [_vlabel(pool[k], k) for k in r["partner1"] if k in pool],
                "p2": [_vlabel(pool[k], k) for k in r["partner2"] if k in pool],
            })

    report_dt = case.approved_at if (final and case.approved_at) else datetime.now(timezone.utc)
    methodology = settings.report_methodology.strip() or f"{T.LAB_METHODOLOGY_UPSTREAM} {T.PIPELINE_METHODOLOGY}"
    show_sig = final and settings.signatures_on_final
    first = case.samples[0] if case.samples else None
    return {
        "lab": settings,
        "title": "Whole Exome Sequencing Analysis",
        "subtitle": SUBTITLES[ctype],
        "case": case,
        "samples": case.samples,
        "role_label": ROLE_LABEL,
        "first": first,
        "report_date": report_dt.astimezone().strftime("%d/%m/%Y"),
        "flag": flag,
        "result_text": RESULT_TEXT.get(rkey, ""),
        "summary_line": review.get("summary_line", ""),
        "sections": sections,
        "at_risk": at_risk,
        "clinical_history": review.get("clinical_history_text") or "",
        "cnv_text": review.get("cnv_text") or "",
        "clinical_interpretation": review.get("clinical_interpretation", ""),
        "recommendations": review.get("recommendations") or [],
        "limitations": review.get("limitations") or [],
        "negative_footnote": T.NEGATIVE_FOOTNOTE,
        "secondary_requested": case.consent_secondary,
        "sf_gene_count": len(ACMG_SF_V3_2),
        "methodology": methodology,
        "seq_total_gb": case.seq_total_gb,
        "seq_q30": case.seq_q30,
        "classification_intro": T.CLASSIFICATION_INTRO,
        "classification_table": T.CLASSIFICATION_TABLE,
        "disclaimer": [d.format(lab=settings.lab_name) for d in T.DISCLAIMER],
        "references": T.REFERENCES,
        "signatories": T.SIGNATORIES,
        "signature_img": _img(settings.signature_image) if show_sig else None,
        "letterhead_img": _img(settings.letterhead_image),
        "gene_table": _gene_table(case, review, sections) if ctype != "couple" else None,
        "final": final,
        "version": case.report_version,
        "mendelian": (case.qc or {}).get("mendelian"),
    }


def render_html(case: Case, final: bool = False) -> str:
    return _env.get_template("report/report.html").render(**build_context(case, final))


def render_pdf(case: Case, final: bool = False) -> Path:
    from weasyprint import HTML  # imported lazily: heavy

    html = render_html(case, final)
    suffix = f"v{case.report_version}" if final else "draft"
    out = settings.reports_dir / f"{case.accession}_{suffix}.pdf"
    HTML(string=html, base_url=str(TEMPLATES)).write_pdf(out)
    return out
