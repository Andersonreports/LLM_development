"""ClinVar CLNSIG parsing. Only the first term used to be read, so compound values
such as "Pathogenic/Likely_pathogenic/Likely_risk_allele" got no category (the
variant lost its ClinVar support) and "Pathogenic&Benign" read as Pathogenic.
The values below are real CLNSIG strings from the 28 Sep 2026 ClinVar release."""
from __future__ import annotations

import pytest

from wes.variants import _clinvar, clinvar_category


@pytest.mark.parametrize("sig, expected", [
    ("Pathogenic", "P"),
    ("Likely_pathogenic", "LP"),
    ("Pathogenic/Likely_pathogenic", "P/LP"),
    ("Pathogenic|risk_factor", "P"),
    ("Pathogenic,_low_penetrance", "P"),
    ("Likely_pathogenic,_low_penetrance", "LP"),
    # previously None: pathogenic records that lost their ClinVar support
    ("Pathogenic/Likely_pathogenic/Likely_risk_allele", "P/LP"),
    ("Likely_pathogenic/Likely_risk_allele", "LP"),
    ("Pathogenic/Likely_risk_allele", "P"),
    ("Uncertain_significance", "VUS"),
    ("Uncertain_significance/Uncertain_risk_allele", "VUS"),
    ("VUS-high", "VUS"),
    ("Likely_benign", "LB"),
    ("Benign", "B"),
    ("Benign/Likely_benign|other", "B"),
    ("Conflicting_classifications_of_pathogenicity", "conflicting"),
    ("Conflicting_classifications_of_pathogenicity|risk_factor", "conflicting"),
    ("Conflicting_interpretations_of_pathogenicity", "conflicting"),
    ("drug_response", None),
    ("risk_factor", None),
    ("not_provided", None),
    ("", None),
])
def test_clinvar_category(sig, expected):
    assert clinvar_category(sig) == expected


def test_pathogenic_and_benign_records_at_one_site_are_a_conflict():
    # VEP joins several ClinVar records at one site with "&"; reading only the first
    # term turned this into a pathogenic call
    assert clinvar_category("Pathogenic&Benign") == "conflicting"
    assert clinvar_category("Likely_benign&Likely_pathogenic") == "conflicting"


def test_compound_pathogenic_value_keeps_stars():
    v = _clinvar({"ClinVar_CLNSIG": "Pathogenic/Likely_pathogenic/Likely_risk_allele",
                  "ClinVar_CLNREVSTAT": "criteria_provided,_multiple_submitters,_no_conflicts"})
    assert (v["clinvar_category"], v["clinvar_stars"]) == ("P/LP", 2)
