"""Regression tests for bugs found by comparing a real tool-generated report
against the lab's actual signed-out report for the same patient (same clinical
history, same VCF): the tool declared POSITIVE with 5 "Likely Pathogenic"
findings in genes unrelated to the phenotype, while the real lab reported
NEGATIVE. Root causes and fixes below."""
from __future__ import annotations

from wes import acmg
from wes.analysis import rule_based_flag
from wes.resources import GeneInfo


def _variant(key, gene, short, points, diseases=None, inheritance_match=True,
            models=("AD",), phenotype_score=0.0, comphet_partners=()):
    return {
        "key": key, "gene": gene, "models": list(models), "comphet_partners": list(comphet_partners),
        "inheritance_match": inheritance_match, "phenotype_score": phenotype_score,
        "rank_score": points / 10, "clinvar_category": None, "clinvar_stars": None,
        "gene_info": {"diseases": diseases or []},
        "acmg": {"short": short, "points": points, "classification": acmg.classify_points([])[1], "criteria": []},
    }


# ---------------------------------------------------------------- Fix A
# A P/LP classification is not, by itself, evidence the gene explains THIS patient's
# presentation. With no HPO terms / indication, the phenotype gate used to accept every
# candidate (score 0 >= threshold 0), so any rare LoF variant anywhere in the exome could
# surface as "POSITIVE - P/LP variant(s) with consistent inheritance and phenotype match".
def test_no_phenotype_never_yields_positive():
    cands = [_variant("v1", "RANDOMGENE", "LP", 9, diseases=[{"name": "Some unrelated disease", "inheritance": ["AD"]}])]
    flag = rule_based_flag("single", cands, None, has_phenotype=False)
    assert flag["flag"] == "INCONCLUSIVE"
    assert "No phenotype information" in flag["reason"]
    assert flag["keys"] == ["v1"]


def test_no_phenotype_with_no_plp_is_negative():
    cands = [_variant("v1", "G", "VUS", 2)]
    flag = rule_based_flag("single", cands, None, has_phenotype=False)
    assert flag["flag"] == "NEGATIVE"


def test_with_phenotype_and_good_match_still_positive():
    # confirms the fix did not break the legitimate, well-supported case
    cands = [_variant("v1", "G", "LP", 9, diseases=[{"name": "Real disease", "inheritance": ["AD"]}],
                      phenotype_score=0.8)]
    flag = rule_based_flag("single", cands, None, has_phenotype=True)
    assert flag["flag"] == "POSITIVE"


# ---------------------------------------------------------------- Fix C
# A gene with literally no known disease association (common for genes VEP will still
# annotate a LoF consequence in) must never reach the primary/explains-the-phenotype
# bucket, even if phenotype data happens to be present and scores nonzero by coincidence.
def test_gene_with_no_known_disease_is_never_positive():
    cands = [_variant("v1", "NODISEASEGENE", "LP", 9, diseases=[], inheritance_match=None, phenotype_score=0.9)]
    flag = rule_based_flag("single", cands, None, has_phenotype=True)
    assert flag["flag"] == "INCONCLUSIVE"
    assert flag["keys"] == ["v1"]


# ---------------------------------------------------------------- Fix B
# VEP sometimes attaches "splice_donor_variant" (the canonical +/-1,2 consequence term) to
# multi-base deletions whose HGVSc notation places them well into the intron (seen on a
# real case: c.147+77_147+121del and c.1097+6_1097+11del both carried this term). PVS1 must
# not assume confirmed canonical disruption just because the term is present.
def test_pvs1_rejects_deep_intronic_labelled_as_canonical_splice():
    g = GeneInfo("TESTGENE", diseases=[], hi_score="3")  # lof_intolerant via ClinGen HI=3
    deep = {"vtype": "lof", "consequence": ["splice_donor_variant", "splice_donor_5th_base_variant"],
            "hgvsc": "c.147+77_147+121del", "exon": None, "spliceai": None}
    crit = acmg._pvs1(deep, g)
    assert crit["strength"] != "VeryStrong"
    assert "not the +/-1,2 dinucleotide" in crit["reason"]


def test_pvs1_accepts_true_canonical_splice_site():
    g = GeneInfo("TESTGENE", diseases=[], hi_score="3")
    canonical = {"vtype": "lof", "consequence": ["splice_donor_variant"],
                "hgvsc": "c.152+1G>A", "exon": None, "spliceai": None}
    crit = acmg._pvs1(canonical, g)
    assert crit["strength"] == "VeryStrong"


def test_pvs1_uses_spliceai_for_non_canonical_with_strong_support():
    g = GeneInfo("TESTGENE", diseases=[], hi_score="3")
    deep = {"vtype": "lof", "consequence": ["splice_donor_variant"],
            "hgvsc": "c.100+6_100+11del", "exon": None, "spliceai": 0.85}
    crit = acmg._pvs1(deep, g)
    assert crit["strength"] == "Strong"


# ---------------------------------------------------------------- negation handling
# "non-consanguineous", "no evidence of X", "denies X" mean the opposite of the matched
# HPO term, not its presence - a real case extracted "consanguineous relationship" as
# present from text that explicitly said "non-consanguineous parents".
def test_negation_is_detected():
    from wes.phenotype import extract_from_text
    from wes.resources import kb
    k = kb()  # the test fixture's small synthetic ontology, not the full downloaded one
    text = "No evidence of myopathy. Denies seizure. Global developmental delay present."
    hits = {h["id"]: h["excluded"] for h in extract_from_text(k, text)}
    assert hits["HP:0003198"] is True   # "no evidence of myopathy"
    assert hits["HP:0001250"] is True   # "denies seizure"
    assert hits["HP:0001263"] is False  # genuinely present


def test_min_intron_offset_parsing():
    assert acmg._min_intron_offset("c.147+77_147+121del") == 77
    assert acmg._min_intron_offset("c.1822-2A>G") == 2
    assert acmg._min_intron_offset("c.39+1G>A") == 1
    assert acmg._min_intron_offset("c.784_787del") is None
    assert acmg._min_intron_offset("c.-29-11C>T") == 11  # UTR position marker not mistaken for offset


def test_pvs1_downgrades_paralog_prone_genes():
    g = GeneInfo("C4B", diseases=[{"name": "Systemic lupus erythematosus", "inheritance": []}])
    v = {"vtype": "lof", "consequence": ["frameshift_variant"], "gene": "C4B",
        "hgvsc": "c.2504del", "exon": "20/41", "spliceai": None}
    crit = acmg._pvs1(v, g)
    assert crit["strength"] == "Supporting"
    assert "near-identical paralog" in crit["reason"]


def test_pvs1_unaffected_for_ordinary_genes():
    g = GeneInfo("TESTGENE", diseases=[], hi_score="3")
    v = {"vtype": "lof", "consequence": ["frameshift_variant"], "gene": "TESTGENE",
        "hgvsc": "c.100del", "exon": "3/10", "spliceai": None}
    crit = acmg._pvs1(v, g)
    assert crit["strength"] == "VeryStrong"


def test_downgraded_splice_call_classifies_below_likely_pathogenic():
    g = GeneInfo("TESTGENE", diseases=[], hi_score="3")
    v = {"vtype": "lof", "consequence": ["splice_donor_variant"], "hgvsc": "c.147+77_147+121del",
        "exon": None, "spliceai": None, "pop_af": None, "pop_af_pop": None, "am_score": None,
        "revel": None, "models": [], "clinvar_category": None, "clinvar_stars": None, "clinvar_sig": None}
    result = acmg.evaluate(v, g, "single", False)
    assert result["short"] != "LP"  # previously reached LP purely from the bogus canonical PVS1
