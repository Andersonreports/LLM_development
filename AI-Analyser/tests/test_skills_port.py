"""Ideas taken from the teammate's skills package (skills - ai - analysis/): a stricter
phenotype fit for variants that are pathogenic only because ClinVar says so, more
tolerant deterministic HPO extraction, and a lab-curated gene-phenotype supplement."""
from __future__ import annotations

import csv

from wes import acmg
from wes.analysis import rule_based_flag
from wes.config import settings
from wes.pipeline import _enforce_phenotype_gate
from wes.resources import CUSTOM_G2P, kb


# ---------------------------------------------------------------- ClinVar-only upgrade
def _clinvar_only(score):
    """Our own evidence says VUS; ClinVar (2*) says Pathogenic."""
    return {
        "key": "v1", "gene": "SCN1A", "models": ["AD"], "comphet_partners": [],
        "inheritance_match": True, "phenotype_score": score, "rank_score": 0.5,
        "vtype": "missense", "clinvar_category": "P", "clinvar_stars": 2,
        "gene_info": {"diseases": [{"name": "Dravet syndrome", "inheritance": ["AD"]}]},
        "acmg": {"short": "VUS", "points": 3, "classification": "Uncertain significance", "criteria": []},
    }


def test_clinvar_only_variant_with_moderate_fit_is_inconclusive_not_positive():
    flag = rule_based_flag("single", [_clinvar_only(0.40)], None, has_phenotype=True, kb=kb())
    assert flag["flag"] == "INCONCLUSIVE"


def test_clinvar_only_variant_with_strong_fit_is_still_positive():
    flag = rule_based_flag("single", [_clinvar_only(0.70)], None, has_phenotype=True, kb=kb())
    assert flag["flag"] == "POSITIVE"


def test_own_evidence_lp_keeps_the_ordinary_threshold():
    v = _clinvar_only(0.40)
    v["acmg"] = {"short": "LP", "points": 7, "classification": "Likely pathogenic", "criteria": []}
    flag = rule_based_flag("single", [v], None, has_phenotype=True, kb=kb())
    assert flag["flag"] == "POSITIVE"


def _gate(score, short="VUS"):
    data = {"result_flag": "POSITIVE", "summary_line": "Dx.", "clinical_interpretation": "Explains it.",
            "reviewer_attention": []}
    v = _clinvar_only(score)
    finding = {"variant_key": "v1", "report_section": "primary", "acmg": {"short": short}}
    fired = _enforce_phenotype_gate(data, [finding], {"v1": v}, {"rule_flag": {"flag": "INCONCLUSIVE"}})
    return data, fired


def test_gate_stops_the_ai_calling_a_clinvar_only_case_positive_on_moderate_fit():
    data, fired = _gate(0.40)
    assert fired and data["result_flag"] == "INCONCLUSIVE"
    assert "ClinVar" in data["reviewer_attention"][0]
    assert f"{settings.min_phenotype_clinvar_upgrade:.2f}" in data["reviewer_attention"][0]


def test_gate_allows_clinvar_only_positive_on_strong_fit():
    data, fired = _gate(0.70)
    assert not fired and data["result_flag"] == "POSITIVE"


def test_gate_uses_ordinary_threshold_when_own_evidence_is_lp():
    data, fired = _gate(0.40, short="LP")
    assert not fired and data["result_flag"] == "POSITIVE"


# ---------------------------------------------------------------- HPO extraction
def _extract(text):
    from wes.phenotype import extract_from_text
    return {h["id"]: h["excluded"] for h in extract_from_text(kb(), text)}


def test_british_spelling_matches_american_hpo_name():
    assert _extract("Raised phenylalanine - hyperphenylalaninaemia on newborn screen.") == {
        "HP:0004923": False}


def test_relative_phenotype_is_excluded():
    hits = _extract("Global developmental delay. Her sister has seizures since infancy.")
    assert hits["HP:0001263"] is False
    assert hits["HP:0001250"] is True


def test_patient_phenotype_after_a_family_sentence_is_present():
    hits = _extract("Mother is well. Seizures from 6 months of age.")
    assert hits["HP:0001250"] is False


def test_existing_negation_still_works():
    hits = _extract("No known history of myopathy. Seizures present.")
    assert hits["HP:0003198"] is True
    assert hits["HP:0001250"] is False


# ---------------------------------------------------------------- curated supplement
def test_tracked_custom_gene_phenotype_file_parses():
    with open(CUSTOM_G2P, encoding="utf-8") as fh:
        reader = csv.DictReader((ln for ln in fh if not ln.lstrip().startswith("#")), delimiter="\t")
        rows = list(reader)
    assert reader.fieldnames[:3] == ["ncbi_gene_id", "gene_symbol", "hpo_id"]
    # every curated line must carry a gene and a well-formed HPO id
    assert all(r["gene_symbol"] and r["hpo_id"].startswith("HP:") for r in rows)
