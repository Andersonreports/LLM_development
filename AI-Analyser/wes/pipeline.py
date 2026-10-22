"""End-to-end case pipeline, executed in a background worker thread."""
from __future__ import annotations

import logging
from datetime import timezone
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import acmg, ai
from .analysis import (carrier_findings, couple_analysis, prioritise, rule_based_flag,
                       secondary_findings)
from .annotate import PipelineError, SampleInput, prepare_case_vcf
from .config import settings
from .db import Case, Event, SessionLocal
from .phenotype import extract_from_text, indication_genes, validate_terms
from .qc import compute_qc
from .resources import kb as load_kb
from .vcf import iter_records, read_header
from .report_text import CNV_NEGATIVE, RECOMMENDATIONS_NEGATIVE
from .variants import clinvar_pathogenic, record_to_variants, worth_keeping

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wes")

ROLES = {"single": ["proband"], "trio": ["proband", "father", "mother"], "couple": ["partner1", "partner2"]}
MAX_STORED_CANDIDATES = 200


def submit(case_id: int, ai_only: bool = False) -> None:
    _executor.submit(run_case, case_id, ai_only)


def _event(db, case: Case, kind: str, msg: str, user: str | None = None) -> None:
    db.add(Event(case_id=case.id, kind=kind, message=msg, user=user))
    db.commit()


# (key, label, what happens) - drives the live pipeline tracker in the UI
STEPS = [
    ("validate", "Validate input", "Checking file format and sample information"),
    ("phenotype", "Phenotype", "Matching the clinical history to known symptom terms"),
    ("annotate", "Normalise & annotate", "Standardising variant calls and adding population frequency, clinical and predicted-impact data"),
    ("parse", "Parse variants", "Reading annotated calls and genotypes"),
    ("qc", "Quality control", "Checking sample quality, genetic sex and relatedness"),
    ("prioritise", "Prioritise & classify", "Ranking variants by inheritance pattern, phenotype fit and classification guidelines"),
    ("ai", "AI interpretation", "Drafting findings, result flag and report text"),
    ("review", "Genome analyst review", "Confirm findings, flag and interpretation"),
    ("approved", "Report issued", "Approved, versioned PDF report"),
]
STEP_LABEL = dict((k, label) for k, label, _ in STEPS)


def _stage(db, case: Case, stage: str) -> None:
    case.stage = stage
    db.commit()
    _event(db, case, "stage", f"[{stage}] {STEP_LABEL[stage]}")


def run_case(case_id: int, ai_only: bool = False) -> None:
    db = SessionLocal()
    case = db.get(Case, case_id)
    if case is None:
        return
    try:
        case.status, case.error = "processing", None
        db.commit()
        if ai_only and case.results:
            _run_ai(db, case)
        else:
            _run_full(db, case)
        case.status, case.stage = "review", "review"
        db.commit()
        _event(db, case, "info", "Analysis complete - ready for scientist review")
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        log.exception("case %s failed", case_id)
        case.status, case.error = "failed", f"{e}"
        db.commit()
        _event(db, case, "error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    finally:
        db.close()


def _identifiers(case: Case) -> list[str]:
    ids = []
    for s in case.samples:
        ids += [x for x in (s.name, s.patient_id) if x]
    return ids


def _run_full(db, case: Case) -> None:
    roles = ROLES[case.case_type]
    work = settings.work_dir / case.accession
    work.mkdir(parents=True, exist_ok=True)
    log_file = work / "pipeline.log"
    kb = load_kb()
    warnings = list(kb.warnings)

    # 1. validate inputs ------------------------------------------------------
    _stage(db, case, "validate")
    inputs = []
    for role in roles:
        s = case.sample(role)
        if s is None:
            raise PipelineError(f"Missing sample for role '{role}'")
        hdr = read_header(Path(s.vcf_path))
        if s.vcf_sample not in hdr.samples:
            raise PipelineError(f"Sample '{s.vcf_sample}' not found in {Path(s.vcf_path).name} (has: {', '.join(hdr.samples)})")
        build = hdr.guess_build()
        if build == "GRCh37":
            raise PipelineError(f"{Path(s.vcf_path).name} appears to be GRCh37/hg19 - this pipeline requires GRCh38")
        if build is None:
            warnings.append(f"{role}: genome build not stated in VCF header - assumed GRCh38")
        inputs.append(SampleInput(role, Path(s.vcf_path), s.vcf_sample))

    # 2. phenotype --------------------------------------------------------------
    _stage(db, case, "phenotype")
    _phenotype(db, case, warnings)
    hpo_ids = [t["id"] for t in case.hpo_terms if not t.get("excluded")]

    # 3. normalise + annotate -------------------------------------------------
    _stage(db, case, "annotate")
    annotated = prepare_case_vcf(inputs, work, log_file)

    # 4. parse --------------------------------------------------------------------
    _stage(db, case, "parse")
    hdr = read_header(annotated)
    rename = {}
    if not set(roles) <= set(hdr.samples):
        rename = {case.sample(r).vcf_sample: r for r in roles}
    sexes = {r: case.sample(r).sex for r in roles}
    affected = {r: case.sample(r).affected for r in roles}
    variants: list[dict] = []
    n_records = 0

    def stream():
        # single pass: QC sees every ALT record, only relevant variants are kept in memory
        nonlocal n_records
        for rec in iter_records(annotated, hdr):
            if rename:
                rec.genotypes = {rename[k]: g for k, g in rec.genotypes.items() if k in rename}
            if not any(g.has_alt for g in rec.genotypes.values()):
                continue
            n_records += 1
            if rec.passed or not settings.require_pass:
                variants.extend(v for v in record_to_variants(rec, sexes) if worth_keeping(v))
            yield rec

    # 5. QC + variant extraction ---------------------------------------------
    _stage(db, case, "qc")
    qc = compute_qc(stream(), roles, sexes, case.case_type)
    case.qc = qc
    db.commit()
    for w in qc["warnings"]:
        _event(db, case, "warning", w)

    # 6. analysis -----------------------------------------------------------------
    _stage(db, case, "prioritise")
    results: dict = {"warnings": warnings,
                     "stats": {"alt_records": n_records, "variant_gene_records_kept": len(variants)}}
    primary_role = "partner1" if case.case_type == "couple" else "proband"

    if case.case_type == "couple":
        couple = couple_analysis(variants, kb, sexes)
        results["couple"] = couple
        results["candidates"] = []
        if case.consent_secondary:
            results["secondary"] = {r: secondary_findings(variants, kb, r, sexes.get(r), "couple") for r in roles}
        flag = rule_based_flag("couple", [], couple, False)
    else:
        parentage = bool(qc.get("parentage_confirmed")) if case.case_type == "trio" else False
        proband_vcf = case.sample("proband").vcf_path
        assumed = {s.role for s in case.samples if s.role in ("father", "mother")
                   and s.vcf_path != proband_vcf} if settings.trio_missing_as_ref else set()
        if assumed:
            warnings.append("Parents supplied as separate VCFs: proband variants absent from a parental VCF are "
                            "treated as reference in that parent (coverage not verified); de novo calls use PM6, not PS2")
        cands = prioritise(variants, case.case_type, kb, hpo_ids, sexes, affected, parentage, assumed)
        results["stats"]["candidates"] = len(cands)
        results["candidates"] = cands[:MAX_STORED_CANDIDATES]
        if case.consent_secondary:
            results["secondary"] = {primary_role: secondary_findings(variants, kb, primary_role,
                                                                     sexes.get(primary_role), case.case_type)}
        if case.consent_carrier:
            results["carrier"] = carrier_findings(variants, kb, primary_role, sexes.get(primary_role), case.case_type)
        flag = rule_based_flag(case.case_type, cands, None, bool(hpo_ids))
    results["gene_list"] = indication_genes(
        kb, hpo_ids, list((case.phenotype_ai or {}).get("suspected_genes") or []),
        settings.gene_list_size, settings.gene_list_min_score) if case.case_type != "couple" else []
    results["rule_flag"] = flag
    case.rule_flag = flag["flag"]
    case.results = results
    db.commit()
    _event(db, case, "info", f"Rule-based flag: {flag['flag']} - {flag['reason']}")

    _run_ai(db, case)


def _phenotype(db, case: Case, warnings: list[str]) -> None:
    kb = load_kb()
    terms = list(case.manual_hpo or [])
    proband = case.proband
    history = (case.clinical_history or "").strip()

    if case.case_type != "couple" and history:
        # deterministic, LLM-independent floor: always run, regardless of whether AI is
        # configured. Free-text clinical history must never go completely unused just
        # because no API key is set - that gap previously let cases through with real
        # clinical text on file but zero phenotype signal reaching the ranking engine.
        local_terms = extract_from_text(kb, history)
        terms += local_terms
        if local_terms:
            _event(db, case, "info",
                  f"{len(local_terms)} phenotype term(s) matched directly from clinical history "
                  f"(no AI): {', '.join(t['name'] for t in local_terms[:6])}" +
                  ("..." if len(local_terms) > 6 else ""))

    if case.case_type != "couple" and history and ai.available():
        try:
            out, usage = ai.extract_phenotype(case.clinical_history, proband.age if proband else None,
                                              proband.sex if proband else None, _identifiers(case))
            case.phenotype_ai = out.model_dump()
            terms += [t.model_dump() for t in out.hpo_terms]
            if out.consanguinity is not None and not case.consanguinity:
                case.consanguinity = "yes" if out.consanguinity else "no"
        except ai.AIError as e:
            warnings.append(f"AI phenotype extraction failed: {e}")
            _event(db, case, "warning", f"AI phenotype extraction failed: {e}")
    elif case.case_type != "couple" and history and not ai.available():
        warnings.append("No AI configured - phenotype terms were matched directly from the clinical "
                        "history text instead (less thorough than AI extraction, but not skipped)")
    elif case.case_type != "couple" and not terms:
        warnings.append("No clinical history or HPO terms were provided - phenotype-driven ranking "
                        "disabled. A result cannot be confirmed as explaining the phenotype without this.")
    case.hpo_terms = validate_terms(kb, terms)
    dropped = len(terms) - len(case.hpo_terms)
    if dropped > 0 and kb.ontology:
        _event(db, case, "info", f"{dropped} HPO term(s) dropped as duplicate or not found in HPO")
    db.commit()


def _all_variants(results: dict) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    for v in results.get("candidates", []):
        pool[v["key"]] = v
    for vs in (results.get("secondary") or {}).values():
        for v in vs:
            pool.setdefault(v["key"], v)
    for v in results.get("carrier") or []:
        pool.setdefault(v["key"], v)
    couple = results.get("couple") or {}
    for v in couple.get("partner1", []) + couple.get("partner2", []):
        pool.setdefault(v["key"], v)
    return pool


def _clinical_context(case: Case) -> dict:
    p = case.proband
    return {
        "age": p.age if p else None,
        "sex": p.sex if p else None,
        "indication": case.indication,
        "summary": (case.phenotype_ai or {}).get("summary") or ai.scrub(case.clinical_history, _identifiers(case))[:3000],
        "hpo_terms": [f"{t['id']} {t['name']}" + (" (excluded)" if t.get("excluded") else "") for t in case.hpo_terms],
        "family_history": ai.scrub(case.family_history or "", _identifiers(case)) or None,
        "consanguinity": case.consanguinity,
        "affected_parents": [s.role for s in case.samples if s.role in ("father", "mother") and s.affected],
        "partner_sexes": {s.role: s.sex for s in case.samples} if case.case_type == "couple" else None,
    }


def _run_ai(db, case: Case) -> None:
    results = case.results or {}
    pool = _all_variants(results)
    if not ai.available():
        _event(db, case, "stage", "[ai:skipped] AI interpretation skipped - AI assistant not configured")
        _event(db, case, "warning", "AI assistant is not configured (no API key set) - a rule-based draft was prepared instead. You can still review and edit everything below.")
        case.ai_result, case.ai_flag = None, None
        case.review = _with_report_fields(case, _with_audit(draft_from_rules(case, pool), pool))
        db.commit()
        return

    _stage(db, case, "ai")
    sec = []
    for vs in (results.get("secondary") or {}).values():
        sec += vs
    context = {
        "case_type": case.case_type,
        "clinical": _clinical_context(case),
        "qc": {"warnings": (case.qc or {}).get("warnings", []),
               "mendelian": (case.qc or {}).get("mendelian"),
               "samples": {r: {k: s.get(k) for k in ("inferred_sex", "declared_sex", "ti_tv", "roh_total_mb")}
                           for r, s in (case.qc or {}).get("samples", {}).items()}},
        "rule_flag": results.get("rule_flag"),
        "candidates": results.get("candidates", [])[: settings.ai_max_candidates],
        "couple": results.get("couple"),
        "secondary": sec,
        "carrier": results.get("carrier"),
    }
    if case.case_type == "couple":
        c = results.get("couple") or {}
        context["candidates"] = c.get("partner1", []) + c.get("partner2", [])
    try:
        out, usage = ai.interpret(context)
    except ai.AIError as e:
        _event(db, case, "warning", f"AI assistant could not complete interpretation ({e}) - a rule-based draft was prepared instead. You can still review and edit everything below.")
        case.review = _with_report_fields(case, _with_audit(draft_from_rules(case, pool), pool))
        db.commit()
        return

    data = out.model_dump()
    attention = list(data["reviewer_attention"])
    findings = []
    for f in data["findings"]:
        v = pool.get(f["variant_key"])
        if v is None:
            attention.append(f"AI referenced unknown variant key {f['variant_key']} - ignored")
            continue
        removed = set(f["criteria_removed"])
        crit = [c for c in v["acmg"]["criteria"] if c["code"] not in removed]
        existing = {c["code"] for c in crit}
        for c in f["criteria_added"]:
            if c["code"] in existing:
                crit = [x for x in crit if x["code"] != c["code"]]
            crit.append(acmg._crit(c["code"], c["strength"], c["reason"], source="ai"))
        merged = acmg.reclassify({**v["acmg"], "criteria": crit})
        if merged["classification"] != f["classification"]:
            attention.append(
                f"{v['gene']} {v.get('hgvsc')}: AI stated '{f['classification']}' but criteria total "
                f"{merged['points']} points = '{merged['classification']}'; points-based class used")
        findings.append({**f, "acmg": merged, "include": f["report_section"] != "not_reported"})
    data["findings"] = findings
    data["reviewer_attention"] = attention
    case.ai_result = data
    case.ai_usage = usage
    case.ai_flag = data["result_flag"]
    case.review = {
        "flag": data["result_flag"],
        "summary_line": data["summary_line"],
        "findings": findings,
        "clinical_interpretation": data["clinical_interpretation"],
        "recommendations": data["recommendations"],
        "limitations": data["limitations"],
        "reviewer_attention": attention,
        "source": "ai",
    }
    case.review = _with_report_fields(case, _with_audit(case.review, pool))
    db.commit()
    _event(db, case, "info", f"AI flag: {data['result_flag']} - {data['flag_rationale']}")
    if case.rule_flag and case.rule_flag != data["result_flag"]:
        _event(db, case, "warning", f"AI flag ({data['result_flag']}) differs from rule-based flag ({case.rule_flag}) - review carefully")


def default_clinical_history(case: Case) -> str:
    p = case.proband
    if case.case_type == "couple":
        names = " and ".join(s.patient_id or s.name for s in case.samples)
        return f"{names} have been evaluated for carrier status of pathogenic variations."
    who = (p.patient_id or p.name) if p else "the patient"
    ind = (case.indication or "").strip().rstrip(".")
    if ind:
        # lower-case the first letter unless it starts an acronym (e.g. "GDD", "DNA")
        ind = ind if len(ind) > 1 and ind[1].isupper() else ind[0].lower() + ind[1:]
        text = f"Proband, {who}, is suspected to be affected with {ind} and has been evaluated for pathogenic variations."
    else:
        text = (case.phenotype_ai or {}).get("summary") or (case.clinical_history or "").strip()
    if case.case_type == "trio":
        text += " Parental samples were analysed to assess segregation of the variants identified."
    return text


def _with_report_fields(case: Case, review: dict) -> dict:
    """Report-level text the reviewer confirms: clinical history, CNV statement."""
    review.setdefault("clinical_history_text", default_clinical_history(case))
    if case.case_type != "couple":
        if review.get("flag") == "NEGATIVE":  # the lab's standard negative-report recommendations
            review["recommendations"] = list(RECOMMENDATIONS_NEGATIVE)
        review.setdefault("cnv_text", CNV_NEGATIVE)
        review["reviewer_attention"] = list(review.get("reviewer_attention", [])) + [
            "CNV statement is the standard template wording. This tool does not call copy-number variants - "
            "confirm CNV analysis was performed, or edit the statement."]
    return review


def _with_audit(review: dict, pool: dict[str, dict]) -> dict:
    """Surface ACMG self-audit violations and segregation cautions of reported variants
    to the reviewer, so they are visible without digging into the candidates table."""
    notes = list(review.get("reviewer_attention", []))
    for f in review.get("findings", []):
        v = pool.get(f["variant_key"], {})
        for a in (v.get("acmg") or {}).get("audit", []):
            notes.append(f"{v.get('gene')} {v.get('hgvsc') or v.get('variant_id')}: {a}")
        for n in v.get("model_notes", []):
            if "CAUTION:" in n:
                caution = n.split("CAUTION:", 1)[1].strip().rstrip(")")
                notes.append(f"{v.get('gene')} {v.get('hgvsc') or v.get('variant_id')}: {caution}")
    return {**review, "reviewer_attention": notes}


def draft_from_rules(case: Case, pool: dict[str, dict]) -> dict:
    rf = (case.results or {}).get("rule_flag") or {"flag": "NEGATIVE", "keys": [], "reason": ""}
    findings = []
    attention: list[str] = []
    section = "primary" if rf["flag"] == "POSITIVE" else ("carrier" if rf["flag"] == "CARRIER" else "vus")
    for k in rf.get("keys", []):
        v = pool.get(k)
        if not v:
            continue
        plp = acmg.is_plp(v) or clinvar_pathogenic(v)
        findings.append(_rule_finding(v, section if plp or section == "vus" else "vus"))
    for vs in ((case.results or {}).get("secondary") or {}).values():
        for v in vs:
            findings.append(_rule_finding(v, "secondary"))
    for v in (case.results or {}).get("carrier") or []:
        findings.append(_rule_finding(v, "carrier"))
    # one entry per variant: the first (most specific) section wins
    seen, unique = set(), []
    for f in findings:
        if f["variant_key"] not in seen:
            seen.add(f["variant_key"])
            unique.append(f)
    for f in unique:
        v = pool[f["variant_key"]]
        if not acmg.is_plp(v) and clinvar_pathogenic(v):
            attention.append(
                f"{v['gene']} {v.get('hgvsc')}: ClinVar {v.get('clinvar_sig')} ({v.get('clinvar_stars')}*) but "
                f"automated criteria total {v['acmg']['points']} points ({v['acmg']['classification']}) - "
                "add the supporting evidence or set the classification before approval")
    return {
        "flag": rf["flag"],
        "summary_line": rf["reason"],
        "findings": unique,
        "clinical_interpretation": "",
        "recommendations": ["Genetic counselling is recommended to discuss these results."],
        "limitations": [],
        "reviewer_attention": attention,
        "source": "rules",
    }


def _rule_finding(v: dict, section: str) -> dict:
    dis = (v.get("gene_info") or {}).get("diseases") or []
    return {
        "variant_key": v["key"], "report_section": section, "classification": v["acmg"]["classification"],
        "criteria_added": [], "criteria_removed": [],
        "disease": "; ".join(d["name"] for d in dis[:2]) or (v.get("clinvar_disease") or ""),
        "disease_inheritance": ", ".join((v.get("gene_info") or {}).get("inheritance", [])),
        "phenotype_fit": "not_applicable", "interpretation": "", "acmg": v["acmg"], "include": True,
    }


def _iso(dt) -> str:
    """SQLite returns naive datetimes that are really UTC; mark them so browsers convert correctly."""
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def progress(case: Case) -> list[dict]:
    """State of every pipeline step for the latest run: done / current / pending /
    skipped / failed, with start time and duration, derived from the event log."""
    events = list(case.events)
    run_start = 0
    for i, e in enumerate(events):
        if e.kind == "info" and e.message in ("Analysis queued", "Re-running AI interpretation"):
            run_start = i
    ai_only = run_start < len(events) and events[run_start].message == "Re-running AI interpretation"
    started, skipped, finished_at = {}, set(), None
    for e in events[run_start:]:
        if e.kind == "stage" and e.message.startswith("["):
            key = e.message[1:e.message.index("]")]
            if key.endswith(":skipped"):
                skipped.add(key.split(":")[0])
            else:
                started[key] = e.at
        elif e.kind == "info" and e.message.startswith("Analysis complete"):
            finished_at = e.at
    order = [k for k, _, _ in STEPS]
    status, cur = case.status, case.stage
    out = []
    for idx, (key, label, desc) in enumerate(STEPS):
        state = "pending"
        if key == "approved":
            state = "done" if status == "approved" else "pending"
        elif key == "review":
            state = "done" if status == "approved" else ("current" if status == "review" else "pending")
        elif key in skipped:
            state = "skipped"
        elif ai_only and order.index(key) < order.index("ai"):
            state = "done"
        elif key in started:
            if status == "failed" and cur == key:
                state = "failed"
            elif status in ("processing", "queued") and cur == key:
                state = "current"
            else:
                state = "done"
        elif status in ("review", "approved") and key not in ("review", "approved"):
            state = "done" if not ai_only else state
        t0 = started.get(key)
        t1 = None
        if t0:
            later = [started[k] for k in order[idx + 1:] if k in started]
            t1 = min(later) if later else (finished_at if state == "done" else None)
        out.append({"key": key, "label": label, "desc": desc, "state": state,
                    "started": _iso(t0) if t0 else None,
                    "seconds": round((t1 - t0).total_seconds(), 1) if t0 and t1 else None})
    return out
