from __future__ import annotations

import time

from conftest import csq, write_vcf

from wes import acmg, ai, pipeline, report
from wes.db import Case, Sample, SessionLocal, init_db

init_db()

SCN1A_FS = csq("A", "frameshift_variant", "HIGH", "SCN1A", exon="5/26", hgvsc="c.664del", hgvsp="p.Arg222fs")
SCN1A_MIS = csq("T", "missense_variant", "MODERATE", "SCN1A", hgvsc="c.2836C>T", hgvsp="p.Arg946Cys", am="0.95")
OTHER_MIS = csq("G", "missense_variant", "MODERATE", "OTHER1", af="0.0001", hgvsc="c.10A>G", am="0.4")
COMMON = csq("C", "missense_variant", "MODERATE", "OTHER1", af="0.2", hgvsc="c.20T>C")
BRCA2_STOP = csq("T", "stop_gained", "HIGH", "BRCA2", exon="11/27", hgvsc="c.5946C>T", hgvsp="p.Ser1982Ter")
PAH_P = csq("A", "missense_variant", "MODERATE", "PAH", af="0.0002", hgvsc="c.1222C>T", hgvsp="p.Arg408Trp",
            clnsig="Pathogenic", clnrev="criteria_provided,_multiple_submitters,_no_conflicts", am="0.99")
PAH_FS = csq("TA", "frameshift_variant", "HIGH", "PAH", exon="3/13", hgvsc="c.300dup", hgvsp="p.Val101fs")
DMD_MIS = csq("A", "missense_variant", "MODERATE", "DMD", hgvsc="c.1000G>A", am="0.95")
DMD_STOP = csq("A", "stop_gained", "HIGH", "DMD", exon="20/79", hgvsc="c.2302C>T", hgvsp="p.Arg768Ter")
CFTR_P = csq("A", "inframe_deletion", "MODERATE", "CFTR", af="0.01", hgvsc="c.1521_1523del", hgvsp="p.Phe508del",
             clnsig="Pathogenic", clnrev="reviewed_by_expert_panel")


def _make_case(db, case_type, samples, **kw) -> Case:
    c = Case(accession=f"T{case_type}{time.time_ns()}", case_type=case_type, clinical_history="", **kw)
    db.add(c)
    db.flush()
    for role, (vcf, col, sex, affected) in samples.items():
        db.add(Sample(case_id=c.id, role=role, name=f"{role} name", vcf_path=str(vcf), vcf_sample=col,
                      sex=sex, affected=affected))
    db.commit()
    db.refresh(c)
    return c


def _reload(case_id):
    db = SessionLocal()
    return db, db.get(Case, case_id)


def test_single_case_positive(tmp_path):
    vcf = write_vcf(tmp_path / "single.vcf", ["S1"], [
        ("2", 1000, "AC", "A", SCN1A_FS, ["0/1"]),
        ("5", 2000, "A", "G", OTHER_MIS, ["0/1"]),
        ("5", 2100, "T", "C", COMMON, ["1/1"]),
        ("13", 3000, "C", "T", BRCA2_STOP, ["0/1"]),
        ("12", 4000, "G", "A", PAH_P, ["0/1"]),
    ])
    db = SessionLocal()
    c = _make_case(db, "single", {"proband": (vcf, "S1", "female", True)},
                   manual_hpo=[{"id": "HP:0001250", "name": ""}, {"id": "HP:0001263", "name": ""}],
                   consent_secondary=True, consent_carrier=True)
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.status == "review", c.error
    cands = c.results["candidates"]
    top = cands[0]
    assert top["gene"] == "SCN1A"
    assert top["models"] == ["AD"]
    assert top["acmg"]["short"] == "LP"
    assert {x["code"] for x in top["acmg"]["criteria"]} >= {"PVS1", "PM2"}
    assert top["phenotype_score"] > 0.5
    assert all(v["gene"] != "OTHER1" or v["pos"] != 2100 for v in cands), "common variant must be filtered"
    assert c.rule_flag == "POSITIVE"
    assert [v["gene"] for v in c.results["secondary"]["proband"]] == ["BRCA2"]
    assert [v["gene"] for v in c.results["carrier"]] == ["PAH"]
    # rule-only draft (AI disabled)
    assert c.review["source"] == "rules"
    sections = {f["report_section"] for f in c.review["findings"]}
    assert {"primary", "secondary", "carrier"} <= sections
    html = report.render_html(c)
    assert "Whole Exome Sequencing Analysis" in html and "SCN1A" in html and "DRAFT" in html
    assert "causative of the reported phenotype was detected" in html
    assert "Appendix 1" in html
    pdf = report.render_pdf(c)
    assert pdf.exists() and pdf.stat().st_size > 5000


def test_trio_comphet_and_de_novo(tmp_path):
    vcf = write_vcf(tmp_path / "trio.vcf", ["KID", "DAD", "MUM"], [
        ("12", 4000, "G", "A", PAH_P, ["0/1", "0/1", "0/0"]),
        ("12", 5000, "T", "TA", PAH_FS, ["0/1", "0/0", "0/1"]),
        ("2", 6000, "C", "T", SCN1A_MIS, ["0/1", "0/0", "0/0"]),
        ("X", 7000000, "G", "A", DMD_MIS, ["1", "0", "0/1"]),
        ("5", 2000, "A", "G", OTHER_MIS, ["0/1", "0/1", "0/0"]),
    ] + [  # common inherited sites so the Mendelian error rate is realistic
        ("6", 10000 + i * 100, "A", "C", COMMON, gts)
        for i, gts in enumerate([["0/1", "0/1", "0/0"], ["1/1", "0/1", "0/1"], ["0/1", "0/0", "1/1"]] * 15)
    ])
    db = SessionLocal()
    c = _make_case(db, "trio", {
        "proband": (vcf, "KID", "male", True), "father": (vcf, "DAD", "male", False),
        "mother": (vcf, "MUM", "female", False)},
        manual_hpo=[{"id": "HP:0004923", "name": ""}, {"id": "HP:0001263", "name": ""}])
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.status == "review", c.error
    by_gene = {}
    for v in c.results["candidates"]:
        by_gene.setdefault(v["gene"], []).append(v)
    pah = by_gene["PAH"]
    assert len(pah) == 2 and all("AR_comphet" in v["models"] for v in pah)
    assert {v["origin"] for v in pah} == {"father", "mother"}
    assert any("in trans" in n for v in pah for n in v["model_notes"])
    fs = next(v for v in pah if v["pos"] == 5000)
    assert "PM3" in {x["code"] for x in fs["acmg"]["criteria"]}
    assert fs["acmg"]["short"] in ("P", "LP")
    scn = by_gene["SCN1A"][0]
    assert scn["models"] == ["de_novo"]
    assert c.qc["mendelian"]["errors"] == 1  # the de novo itself
    assert c.qc["parentage_confirmed"]
    assert "PS2" in {x["code"] for x in scn["acmg"]["criteria"]}
    dmd = by_gene["DMD"][0]
    assert dmd["models"] == ["XLR"] and dmd["zygosity"]["proband"] == "hemizygous"
    # OTHER1 inherited from unaffected father - dominant only with reduced penetrance
    other = by_gene["OTHER1"][0]
    assert other["models"] == ["AD_inherited"]
    assert c.rule_flag == "POSITIVE"
    html = report.render_html(c)
    assert "Trio analysis" in html and "Family member" in html


def test_couple_combined_vcf(tmp_path):
    vcf = write_vcf(tmp_path / "couple.vcf", ["WIFE", "HUSB"], [
        ("7", 117559590, "ATCT", "A", CFTR_P, ["0/1", "0/1"]),
        ("X", 31000000, "G", "A", DMD_STOP, ["0/1", "0"]),
        ("12", 4000, "G", "A", PAH_P, ["0/0", "0/1"]),
    ])
    db = SessionLocal()
    c = _make_case(db, "couple", {"partner1": (vcf, "WIFE", "female", False),
                                  "partner2": (vcf, "HUSB", "male", False)})
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.status == "review", c.error
    couple = c.results["couple"]
    assert {r["gene"] for r in couple["at_risk"]} == {"CFTR", "DMD"}
    assert {v["gene"] for v in couple["partner2"]} == {"CFTR", "PAH"}
    assert c.rule_flag == "POSITIVE"
    html = report.render_html(c)
    assert "Increased reproductive risk" in html and "Couple carrier screening" in html and "Partner 2" in html


def test_couple_carrier_only(tmp_path):
    vcf = write_vcf(tmp_path / "couple2.vcf", ["A", "B"], [
        ("12", 4000, "G", "A", PAH_P, ["0/0", "0/1"]),
    ])
    db = SessionLocal()
    c = _make_case(db, "couple", {"partner1": (vcf, "A", "female", False), "partner2": (vcf, "B", "male", False)})
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.rule_flag == "CARRIER"
    assert "Carrier status was detected" in report.render_html(c)


def test_points_classification():
    crit = [acmg._crit("PVS1", "VeryStrong", ""), acmg._crit("PM2", "Supporting", "")]
    assert acmg.classify_points(crit) == (9, "Likely pathogenic")
    crit.append(acmg._crit("PP4", "Supporting", ""))
    assert acmg.classify_points(crit) == (10, "Pathogenic")
    assert acmg.classify_points([acmg._crit("BA1", "StandAlone", "")])[1] == "Benign"
    assert acmg.classify_points([acmg._crit("BP4", "Supporting", ""), acmg._crit("BS1", "Strong", "")])[1] == "Likely benign"


def test_ai_merge(tmp_path, monkeypatch):
    vcf = write_vcf(tmp_path / "ai.vcf", ["S1"], [("2", 1000, "AC", "A", SCN1A_FS, ["0/1"])])
    db = SessionLocal()
    c = _make_case(db, "single", {"proband": (vcf, "S1", "female", True)},
                   manual_hpo=[{"id": "HP:0001250", "name": ""}])
    key = None

    def fake_interpret(context):
        nonlocal key
        key = context["candidates"][0]["key"]
        return ai.InterpretationOut(
            result_flag="POSITIVE", flag_rationale="LP SCN1A null variant fits Dravet phenotype",
            summary_line="Heterozygous likely pathogenic SCN1A variant identified",
            findings=[ai.Finding(
                variant_key=key, report_section="primary", classification="Pathogenic",
                criteria_added=[ai.Criterion(code="PP4", strength="Supporting", reason="Phenotype specific")],
                criteria_removed=[], disease="Dravet syndrome", disease_inheritance="AD",
                phenotype_fit="strong", interpretation="This variant ...")],
            excluded_candidates=[], clinical_interpretation="Result explains phenotype.",
            recommendations=["Genetic counselling"], limitations=[], reviewer_attention=[]), {"model": "fake"}

    monkeypatch.setattr(ai, "available", lambda: True)
    monkeypatch.setattr(ai, "interpret", fake_interpret)
    monkeypatch.setattr(ai, "extract_phenotype", lambda *a, **k: (_ for _ in ()).throw(ai.AIError("skip")))
    c.clinical_history = "seizures"
    db.commit()
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.status == "review", c.error
    assert c.ai_flag == "POSITIVE"
    f = c.review["findings"][0]
    assert f["acmg"]["classification"] == "Pathogenic"   # 9 + PP4 = 10
    assert any(x["source"] == "ai" and x["code"] == "PP4" for x in f["acmg"]["criteria"])
    assert c.review["source"] == "ai"


def test_negative_report_matches_lab_template(tmp_path):
    vcf = write_vcf(tmp_path / "neg.vcf", ["S1"], [("5", 2100, "T", "C", COMMON, ["1/1"])])
    db = SessionLocal()
    c = _make_case(db, "single", {"proband": (vcf, "S1", "female", True)},
                   hospital="CMC", referring_clinician="Dr A", indication="Carcinoma right breast",
                   seq_total_gb="12.6 GB", seq_q30="96.34 %",
                   manual_hpo=[{"id": "HP:0001250", "name": ""}])
    c.samples[0].patient_id, c.samples[0].pin, c.samples[0].age = "MEL - 16564", "ADK0000005657", "52"
    db.commit()
    pipeline.run_case(c.id)
    db, c = _reload(c.id)
    assert c.status == "review", c.error
    assert c.rule_flag == "NEGATIVE"
    html = report.render_html(c)
    for needle in ["Patient ID: MEL - 16564", "PIN: ADK0000005657", "Age: 52 Years",
                   "Proband, MEL - 16564, is suspected to be affected with carcinoma right breast",
                   "No pathogenic or likely pathogenic variant(s) causative of the reported phenotype was detected",
                   "No other variant that warrants to be reported", "CNV Findings",
                   "Alternative test is strongly recommended to rule out the deletion/duplication.",
                   "Genetic counselling is advised.", "Total Read Generated", "12.6 GB", "96.34 %",
                   "Variant of Uncertain Significance", "Disclaimer", "References",
                   "This report has been reviewed and approved by"]:
        assert needle in html, needle
    assert "data:image/jpeg" not in html          # signatures only on approved reports
    c.status, c.report_version = "approved", 1
    final = report.render_html(c, final=True)
    assert "data:image/jpeg" in final and "DRAFT" not in final.split("<body>")[1][:300]
    assert report.render_pdf(c).stat().st_size > 20000
