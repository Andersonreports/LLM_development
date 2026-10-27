"""POSITIVE asserts that a variant explains the patient. The AI does not reliably respect
the phenotype gate even when told to, so the pipeline enforces it; these tests pin that
behaviour, and the normalisation of criterion codes the AI sends."""
from __future__ import annotations

import pytest

from wes import acmg
from wes.config import settings
from wes.pipeline import _enforce_phenotype_gate


def _variant(key, gene, score):
    return {"key": key, "gene": gene, "phenotype_score": score}


def _finding(key, section="primary"):
    return {"variant_key": key, "report_section": section}


def _data(flag="POSITIVE", summary="Diagnosis found.", interp="The variant explains it."):
    return {"result_flag": flag, "summary_line": summary, "clinical_interpretation": interp,
            "reviewer_attention": ["pre-existing note"]}


def _case(score, flag="POSITIVE", section="primary", rule_flag="NEGATIVE"):
    data = _data(flag)
    fired = _enforce_phenotype_gate(data, [_finding("v1", section)],
                                    {"v1": _variant("v1", "CENPJ", score)},
                                    {"rule_flag": {"flag": rule_flag}})
    return data, fired


def test_positive_without_phenotype_match_is_downgraded():
    data, fired = _case(0.06)
    assert fired and data["result_flag"] == "NEGATIVE"   # the rule engine's verdict
    assert "0.06" in data["reviewer_attention"][0] and "CENPJ" in data["reviewer_attention"][0]
    assert "set the flag back to POSITIVE" in data["reviewer_attention"][0]


def test_the_narrative_is_withdrawn_not_left_contradicting_the_flag():
    data = _data()
    _enforce_phenotype_gate(data, [_finding("v1")], {"v1": _variant("v1", "CENPJ", 0.06)},
                            {"rule_flag": {"flag": "NEGATIVE", "reason": "No P/LP variants identified"}})
    # the AI's argument for a diagnosis must not survive into the report
    assert data["clinical_interpretation"] == ""
    assert "Diagnosis found." not in data["summary_line"]
    # but the banner needs a line that matches the flag, not an empty space
    assert data["summary_line"] == "No P/LP variants identified"
    notes = " ".join(data["reviewer_attention"])
    assert "Diagnosis found." in notes and "The variant explains it." in notes
    assert "pre-existing note" in notes          # earlier notes are kept


def test_positive_with_a_real_phenotype_match_is_untouched():
    data, fired = _case(0.72)
    assert not fired and data["result_flag"] == "POSITIVE"
    assert data["summary_line"] == "Diagnosis found."
    assert data["reviewer_attention"] == ["pre-existing note"]


def test_score_exactly_at_the_threshold_is_accepted():
    data, fired = _case(settings.min_phenotype_for_positive)
    assert not fired and data["result_flag"] == "POSITIVE"


@pytest.mark.parametrize("flag", ["NEGATIVE", "INCONCLUSIVE", "CARRIER"])
def test_other_flags_are_never_touched(flag):
    data, fired = _case(0.01, flag=flag)
    assert not fired and data["result_flag"] == flag and data["summary_line"]


def test_variants_not_in_the_report_do_not_count():
    """A dismissed variant cannot prop up a POSITIVE flag."""
    data = _data()
    pool = {"a": _variant("a", "SCN1A", 0.9), "b": _variant("b", "CENPJ", 0.05)}
    findings = [_finding("a", "not_reported"), _finding("b", "primary")]
    assert _enforce_phenotype_gate(data, findings, pool, {"rule_flag": {"flag": "INCONCLUSIVE"}})
    assert data["result_flag"] == "INCONCLUSIVE" and "0.05" in data["reviewer_attention"][0]


@pytest.mark.parametrize("section", ["carrier", "secondary"])
def test_carrier_and_secondary_findings_cannot_make_a_case_positive(section):
    """What the real CENPJ case did: POSITIVE with everything filed as a carrier finding."""
    data = _data()
    assert _enforce_phenotype_gate(data, [_finding("s", section)], {"s": _variant("s", "BRCA2", 0.9)},
                                   {"rule_flag": {"flag": "NEGATIVE"}})
    assert data["result_flag"] == "NEGATIVE"
    assert "primary or VUS section" in data["reviewer_attention"][0]


# ---------------------------------------------------------------- criterion codes
@pytest.mark.parametrize("raw,expected", [
    ("PVS1", "PVS1"),
    ("PVS1_VeryStrong", "PVS1"),      # what the model actually sent on the real case
    ("pvs1_verystrong", "PVS1"),
    ("PM2_P", "PM2"),
    ("pp3", "PP3"),
    ("  BA1 ", "BA1"),
    ("PM3-Moderate", "PM3"),
    ("PVS2", None),                   # not a real criterion
    ("PP9", None),
    ("", None),
    ("explains the phenotype", None),
])
def test_criterion_codes_are_normalised(raw, expected):
    assert acmg.normalise_code(raw) == expected


def test_every_code_the_engine_emits_survives_normalisation():
    for code in acmg.VALID_CODES:
        assert acmg.normalise_code(code) == code
