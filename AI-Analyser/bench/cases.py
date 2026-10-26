"""Synthetic interpretation cases with known-correct answers.

Each case is realistic pipeline output (the same shape ai.interpret() receives) plus an
`expect` block describing what a competent interpretation must produce. No patient data:
these are hand-built from textbook gene-disease relationships so they can live in git and
be re-run against any model.
"""
from __future__ import annotations


from wes import acmg

# Build criteria and scores with the real engine, so a case can never disagree with
# itself (hand-written point totals drift from the scoring rules)
_crit = acmg._crit


def _acmg(criteria):
    pts, cls = acmg.classify_points(criteria)
    return {"criteria": criteria, "points": pts, "classification": cls, "short": acmg.SHORT[cls],
            "audit": []}


def _gt(dp=45, gq=99, ab=0.48):
    return {"dp": dp, "gq": gq, "ab": ab}


def _qc(role="proband", sex="female"):
    return {"warnings": [], "mendelian": None,
            "samples": {role: {"inferred_sex": sex, "declared_sex": sex, "ti_tv": 2.4,
                               "roh_total_mb": 11.0}}}


# --------------------------------------------------------------- 1. dominant de novo
SCN1A = {
    "key": "2-166848646-C-T:SCN1A", "variant_id": "2-166848646-C-T", "gene": "SCN1A",
    "transcript": "NM_001165963.4", "hgvsc": "c.664C>T", "hgvsp": "p.Arg222Ter",
    "consequence_str": "stop gained", "exon": "5/26", "zygosity": {"proband": "het"},
    "models": ["de_novo"], "model_notes": ["Heterozygous, absent in both parents (de novo)"],
    "origin": "de_novo", "pop_af": None,
    "gene_info": {"diseases": [{"name": "Dravet syndrome", "inheritance": ["AD"]},
                               {"name": "Generalized epilepsy with febrile seizures plus",
                                "inheritance": ["AD"]}],
                  "inheritance": ["AD"], "loeuf": 0.11, "clingen_hi": "3"},
    "phenotype_score": 0.81,
    "phenotype_matches": [{"patient_name": "Seizure", "match_name": "Seizure"},
                          {"patient_name": "Global developmental delay", "match_name": "Intellectual disability"}],
    "acmg": _acmg([_crit("PVS1", "VeryStrong", "Predicted null variant (NMD expected); LoF is a known disease mechanism"),
                   _crit("PS2", "Strong", "De novo with confirmed parentage (trio Mendelian QC passed)"),
                   _crit("PM2", "Supporting", "Absent from gnomAD exomes/genomes")]),
    "genotypes": {"proband": _gt(), "father": _gt(40, 99, 0.0), "mother": _gt(44, 99, 0.0)},
}
DISTRACTOR = {
    "key": "5-140100000-A-G:PCDHB4", "variant_id": "5-140100000-A-G", "gene": "PCDHB4",
    "hgvsc": "c.310A>G", "hgvsp": "p.Lys104Glu", "consequence_str": "missense variant",
    "zygosity": {"proband": "het"}, "models": ["AD"], "pop_af": 0.00008,
    "gene_info": {"diseases": [], "inheritance": [], "loeuf": 1.42},
    "phenotype_score": 0.0, "phenotype_matches": [],
    "acmg": _acmg([_crit("PM2", "Supporting", "Very rare in gnomAD (0.0080%)")]),
    "genotypes": {"proband": _gt(30, 80, 0.51)},
}

# --------------------------------------------------------------- 2. recessive compound het
PAH_A = {
    "key": "12-102852882-G-A:PAH", "variant_id": "12-102852882-G-A", "gene": "PAH",
    "transcript": "NM_000277.3", "hgvsc": "c.1222C>T", "hgvsp": "p.Arg408Trp",
    "consequence_str": "missense variant", "exon": "12/13", "zygosity": {"proband": "het"},
    "models": ["AR_comphet"], "comphet_partners": ["12-102912801-T-C:PAH"],
    "model_notes": ["Compound heterozygous with c.1066-11G>A (in trans, one allele from each parent)"],
    "origin": "father", "pop_af": 0.00008, "clinvar_sig": "Pathogenic", "clinvar_stars": 3,
    "gene_info": {"diseases": [{"name": "Phenylketonuria", "inheritance": ["AR"]}],
                  "inheritance": ["AR"], "loeuf": 0.95},
    "phenotype_score": 0.74,
    "phenotype_matches": [{"patient_name": "Hyperphenylalaninemia", "match_name": "Hyperphenylalaninemia"}],
    "acmg": _acmg([_crit("PM3", "Moderate", "Observed in trans with a pathogenic variant"),
                   _crit("PP3", "Moderate", "REVEL 0.871 (>=0.773)"),
                   _crit("PM2", "Supporting", "Very rare in gnomAD (0.0080%)"),
                   _crit("PP5", "Strong", "ClinVar Pathogenic (3*)")]),
    "genotypes": {"proband": _gt(), "father": _gt(), "mother": _gt(42, 99, 0.0)},
}
PAH_B = {
    "key": "12-102912801-T-C:PAH", "variant_id": "12-102912801-T-C", "gene": "PAH",
    "transcript": "NM_000277.3", "hgvsc": "c.1066-11G>A", "consequence_str": "splice region variant, intron variant",
    "intron": "10/12", "zygosity": {"proband": "het"}, "models": ["AR_comphet"],
    "comphet_partners": ["12-102852882-G-A:PAH"],
    "model_notes": ["Compound heterozygous with c.1222C>T (in trans, one allele from each parent)"],
    "origin": "mother", "pop_af": 0.0004, "clinvar_sig": "Pathogenic", "clinvar_stars": 2,
    "spliceai": 0.92,
    "gene_info": {"diseases": [{"name": "Phenylketonuria", "inheritance": ["AR"]}],
                  "inheritance": ["AR"], "loeuf": 0.95},
    "phenotype_score": 0.74,
    "phenotype_matches": [{"patient_name": "Hyperphenylalaninemia", "match_name": "Hyperphenylalaninemia"}],
    "acmg": _acmg([_crit("PM3", "Moderate", "Observed in trans with a pathogenic variant"),
                   _crit("PP3", "Moderate", "SpliceAI delta score 0.92 (>=0.2), strong splice prediction"),
                   _crit("PM2", "Supporting", "Very rare in gnomAD (0.0400%)"),
                   _crit("PP5", "Supporting", "ClinVar Pathogenic (2*)")]),
    "genotypes": {"proband": _gt(), "father": _gt(38, 99, 0.0), "mother": _gt()},
}

# --------------------------------------------------------------- 3. negative
UNRELATED = {
    "key": "19-45411941-T-C:APOE", "variant_id": "19-45411941-T-C", "gene": "APOE",
    "hgvsc": "c.388T>C", "hgvsp": "p.Cys130Arg", "consequence_str": "missense variant",
    "zygosity": {"proband": "het"}, "models": ["AD"], "pop_af": 0.0032,
    "gene_info": {"diseases": [{"name": "Alzheimer disease, susceptibility to", "inheritance": []}],
                  "inheritance": [], "loeuf": 1.1},
    "phenotype_score": 0.04, "phenotype_matches": [],
    "acmg": _acmg([]),
    "genotypes": {"proband": _gt(36, 95, 0.49)},
}

# --------------------------------------------------------------- 4. VUS, inconclusive
MYH7 = {
    "key": "14-23424090-C-T:MYH7", "variant_id": "14-23424090-C-T", "gene": "MYH7",
    "transcript": "NM_000257.4", "hgvsc": "c.2156G>A", "hgvsp": "p.Arg719Gln",
    "consequence_str": "missense variant", "exon": "19/40", "zygosity": {"proband": "het"},
    "models": ["AD_inherited"], "origin": "mother", "pop_af": 0.000006,
    "model_notes": ["Inherited from unaffected mother - causal only with reduced penetrance/variable expressivity"],
    "clinvar_sig": "Uncertain significance", "clinvar_stars": 1, "revel": 0.69,
    "gene_info": {"diseases": [{"name": "Hypertrophic cardiomyopathy", "inheritance": ["AD"]}],
                  "inheritance": ["AD"], "loeuf": 0.31, "clingen_hi": "0"},
    "phenotype_score": 0.62,
    "phenotype_matches": [{"patient_name": "Hypertrophic cardiomyopathy", "match_name": "Hypertrophic cardiomyopathy"}],
    "acmg": _acmg([_crit("PM2", "Supporting", "Very rare in gnomAD (0.0006%)"),
                   _crit("PP3", "Supporting", "REVEL 0.690 (>=0.644)"),
                   _crit("PP2", "Supporting", "Missense-constrained gene (Z=3.41)")]),
    "genotypes": {"proband": _gt(52, 99, 0.46)},
}


CASES = [
    {
        "name": "dominant_de_novo",
        "description": "SCN1A nonsense, de novo, Dravet-like phenotype -> clear POSITIVE",
        "context": {
            "case_type": "trio",
            "clinical": {"age": "3", "sex": "female", "indication": "Epilepsy with developmental delay",
                         "summary": "Three-year-old girl with febrile seizures from 6 months progressing to "
                                    "prolonged generalised tonic-clonic seizures, and global developmental "
                                    "delay. Non-consanguineous parents, no family history of epilepsy.",
                         "hpo_terms": ["HP:0001250 Seizure", "HP:0001263 Global developmental delay",
                                       "HP:0002373 Febrile seizure"],
                         "family_history": None, "consanguinity": "no", "affected_parents": [],
                         "partner_sexes": None},
            "qc": _qc(), "rule_flag": {"flag": "POSITIVE", "reason": "P/LP variant with consistent inheritance "
                                                                    "and phenotype match", "keys": [SCN1A["key"]]},
            "candidates": [SCN1A, DISTRACTOR], "couple": None, "secondary": [], "carrier": None,
        },
        "expect": {
            "flag": "POSITIVE",
            "primary": [SCN1A["key"]],
            "not_primary": [DISTRACTOR["key"]],
            # SCN1A is on chromosome 2 and the supplied inheritance is AD
            "forbidden": ["x-link", "x link", "x-inactivation", "x inactivation", "lyonization",
                          "hemizyg", "autosomal recessive", "biallelic"],
        },
    },
    {
        "name": "recessive_comphet",
        "description": "PAH compound heterozygous in trans, PKU phenotype -> POSITIVE, both alleles reported",
        "context": {
            "case_type": "trio",
            "clinical": {"age": "2", "sex": "male", "indication": "Hyperphenylalaninaemia on newborn screening",
                         "summary": "Two-year-old boy with persistently raised blood phenylalanine detected on "
                                    "newborn screening, and mild developmental delay. Parents unaffected.",
                         "hpo_terms": ["HP:0004923 Hyperphenylalaninemia", "HP:0001263 Global developmental delay"],
                         "family_history": None, "consanguinity": "no", "affected_parents": [],
                         "partner_sexes": None},
            "qc": _qc(sex="male"),
            "rule_flag": {"flag": "POSITIVE", "reason": "P/LP variant with consistent inheritance and phenotype match",
                          "keys": [PAH_A["key"]]},
            "candidates": [PAH_A, PAH_B], "couple": None, "secondary": [], "carrier": None,
        },
        "expect": {
            "flag": "POSITIVE",
            "primary": [PAH_A["key"], PAH_B["key"]],
            "not_primary": [],
            "forbidden": ["x-link", "x inactivation", "autosomal dominant", "de novo"],
        },
    },
    {
        "name": "negative",
        "description": "Only a common, phenotype-irrelevant VUS -> NEGATIVE, nothing reported",
        "context": {
            "case_type": "single",
            "clinical": {"age": "5", "sex": "male", "indication": "Short stature",
                         "summary": "Five-year-old boy referred for short stature below the third centile, "
                                    "otherwise well, normal development and no dysmorphic features.",
                         "hpo_terms": ["HP:0004322 Short stature"],
                         "family_history": None, "consanguinity": "no", "affected_parents": [],
                         "partner_sexes": None},
            "qc": _qc(sex="male"),
            "rule_flag": {"flag": "NEGATIVE", "reason": "No P/LP or phenotype-relevant VUS identified", "keys": []},
            "candidates": [UNRELATED], "couple": None, "secondary": [], "carrier": None,
        },
        "expect": {
            "flag": "NEGATIVE",
            "primary": [],
            "not_primary": [UNRELATED["key"]],
            "forbidden": ["pathogenic variant was identified", "explains the phenotype"],
        },
    },
    {
        "name": "vus_inconclusive",
        "description": "MYH7 VUS in a phenotype-matching gene, inherited from an unaffected parent -> INCONCLUSIVE",
        "context": {
            "case_type": "single",
            "clinical": {"age": "14", "sex": "male", "indication": "Hypertrophic cardiomyopathy",
                         "summary": "Fourteen-year-old boy with asymmetric septal hypertrophy on echocardiography "
                                    "and exertional breathlessness. Mother reportedly unaffected on recent echo.",
                         "hpo_terms": ["HP:0001639 Hypertrophic cardiomyopathy", "HP:0002094 Dyspnea"],
                         "family_history": "Mother had a normal echocardiogram last year.",
                         "consanguinity": "no", "affected_parents": [], "partner_sexes": None},
            "qc": _qc(sex="male"),
            "rule_flag": {"flag": "INCONCLUSIVE",
                          "reason": "Variant(s) of uncertain significance in phenotype-relevant gene(s)",
                          "keys": [MYH7["key"]]},
            "candidates": [MYH7], "couple": None, "secondary": [], "carrier": None,
        },
        "expect": {
            "flag": "INCONCLUSIVE",
            "primary": [],                 # a VUS belongs in the "vus" section, not "primary"
            "not_primary": [MYH7["key"]],
            "forbidden": ["x-link", "confirms the diagnosis", "definitive diagnosis"],
        },
    },
]

BY_NAME = {c["name"]: c for c in CASES}
