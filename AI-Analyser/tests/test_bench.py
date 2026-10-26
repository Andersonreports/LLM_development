"""The benchmark must be self-consistent and must score correctly, otherwise its
verdicts on a model are meaningless."""
from __future__ import annotations

import pytest

from bench.cases import CASES
from bench.run import CHECKS, forbidden_hits, score
from wes import acmg


def _finding(key, section, classification, added=(), removed=()):
    return {"variant_key": key, "report_section": section, "classification": classification,
            "criteria_added": list(added), "criteria_removed": list(removed),
            "disease": "", "disease_inheritance": "", "phenotype_fit": "strong", "interpretation": ""}


def _output(case, **kw):
    """A model answer that gets this case completely right."""
    pool = {v["key"]: v for v in case["context"]["candidates"]}
    base = {
        "result_flag": case["expect"]["flag"], "flag_rationale": "", "summary_line": "",
        "findings": [_finding(k, "primary", pool[k]["acmg"]["classification"])
                     for k in case["expect"]["primary"]],
        "excluded_candidates": list(case["expect"]["not_primary"]),
        "clinical_interpretation": "", "recommendations": [], "limitations": [], "reviewer_attention": [],
    }
    return base | kw


def test_every_case_is_internally_consistent():
    """Stated points and classification must match what the ACMG engine computes."""
    for case in CASES:
        for v in case["context"]["candidates"]:
            pts, cls = acmg.classify_points(v["acmg"]["criteria"])
            assert (v["acmg"]["points"], v["acmg"]["classification"]) == (pts, cls), v["key"]


def test_expected_answers_reference_real_variants():
    for case in CASES:
        keys = {v["key"] for v in case["context"]["candidates"]}
        assert set(case["expect"]["primary"]) <= keys, case["name"]
        assert set(case["expect"]["not_primary"]) <= keys, case["name"]
        assert case["expect"]["flag"] in ("POSITIVE", "INCONCLUSIVE", "NEGATIVE", "CARRIER")


def test_a_perfect_answer_scores_full_marks():
    for case in CASES:
        res = score(case, _output(case))
        assert all(res[c] for c in CHECKS), (case["name"], res["_detail"])


def test_each_check_catches_its_own_failure():
    case = next(c for c in CASES if c["name"] == "dominant_de_novo")
    key = case["expect"]["primary"][0]

    assert not score(case, _output(case, result_flag="NEGATIVE"))["flag"]
    assert not score(case, _output(case, findings=[]))["primary"]

    removed = [_finding(key, "primary", "Pathogenic", removed=["PVS1"])]
    assert not score(case, _output(case, findings=removed))["no_removal"]

    # claims Pathogenic while deleting the criterion that justifies it
    assert not score(case, _output(case, findings=removed))["consistent"]

    fabricated = _output(case, clinical_interpretation="Due to skewed X-inactivation in this female.")
    assert not score(case, fabricated)["grounded"]


@pytest.mark.parametrize("text,expected", [
    # asserted - must be flagged
    ("SCN1A is classically autosomal recessive for Dravet syndrome.", ["autosomal recessive"]),
    ("This finding provides a definitive diagnosis.", ["definitive diagnosis"]),
    # negated or contrasted - correct clinical language, must NOT be flagged
    ("The unaffected status of the mother precludes a definitive diagnosis.", []),
    ("This does not provide a definitive diagnosis.", []),
    ("The gene is autosomal dominant, not autosomal recessive.", []),
    ("Unlike autosomal recessive disease, a single allele suffices here.", []),
    # negation in an earlier sentence must not clear a later assertion
    ("No family history was reported. The gene is autosomal recessive.", ["autosomal recessive"]),
])
def test_forbidden_phrases_respect_negation(text, expected):
    assert forbidden_hits(text.lower(), ["autosomal recessive", "definitive diagnosis"]) == expected
