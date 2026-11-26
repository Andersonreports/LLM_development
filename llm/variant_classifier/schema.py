"""The ordinal label and the feature columns - the single source of truth both the
synthetic generator and the real-data loaders conform to."""
from __future__ import annotations

# ---- the ordered 5-tier label (ACMG) ---------------------------------------
# Index IS the ordinal value; distance between indices is what the model is scored on.
TIERS = ["B", "LB", "VUS", "LP", "P"]
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}
N_TIERS = len(TIERS)
TIER_FULL = {
    "B": "Benign", "LB": "Likely benign", "VUS": "Uncertain significance",
    "LP": "Likely pathogenic", "P": "Pathogenic",
}

# ClinVar CLNSIG -> tier. Combination calls (e.g. "Benign/Likely_benign") map to the
# milder of the two, matching how they are usually reported. "Conflicting" is dropped.
CLINVAR_TO_TIER = {
    "benign": "B",
    "benign/likely_benign": "B",
    "likely_benign": "LB",
    "uncertain_significance": "VUS",
    "uncertain significance": "VUS",
    "likely_pathogenic": "LP",
    "pathogenic/likely_pathogenic": "LP",
    "likely_pathogenic/pathogenic": "LP",
    "pathogenic": "P",
}
# ClinVar review status (CLNREVSTAT) worth at least ~1 star; "no_assertion"/"no_criteria"
# and single-submitter-conflicting are excluded by default.
CLINVAR_REVSTAT_OK = {
    "criteria_provided,_single_submitter",
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}

# ---- features --------------------------------------------------------------
# All numeric so HistGradientBoosting can take them directly (it handles NaN natively).
# "higher_is_pathogenic" documents the expected direction; missingness is expected and fine.
FEATURES: list[str] = [
    "gnomad_af",          # population allele frequency (lower -> more likely pathogenic)
    "revel",              # 0..1 meta-predictor
    "cadd_phred",         # ~0..40
    "spliceai",           # 0..1 max delta
    "gerp",               # conservation
    "phylop",             # conservation
    "consequence_severity",  # engineered: 0 synonymous .. 1 nonsense/frameshift
    "loeuf",              # gene constraint (LOWER -> more constrained -> pathogenic-prone)
    "missense_z",         # gene missense constraint
]

# Severity rank for the worst VEP consequence of a variant, scaled to 0..1.
CONSEQUENCE_SEVERITY = {
    "synonymous_variant": 0.05,
    "intron_variant": 0.05,
    "5_prime_utr_variant": 0.1,
    "3_prime_utr_variant": 0.1,
    "missense_variant": 0.5,
    "inframe_deletion": 0.45,
    "inframe_insertion": 0.45,
    "splice_region_variant": 0.6,
    "splice_donor_variant": 0.9,
    "splice_acceptor_variant": 0.9,
    "stop_gained": 1.0,
    "frameshift_variant": 1.0,
    "start_lost": 0.9,
    "stop_lost": 0.85,
}


def consequence_severity(consequence: str | None) -> float:
    """Map a VEP consequence string (possibly '&'-joined) to its worst severity in 0..1."""
    if not consequence:
        return float("nan")
    worst = 0.0
    seen = False
    for part in str(consequence).split("&"):
        if part in CONSEQUENCE_SEVERITY:
            worst = max(worst, CONSEQUENCE_SEVERITY[part])
            seen = True
    return worst if seen else float("nan")
