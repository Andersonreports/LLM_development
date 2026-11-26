"""Official ACMG/AMP 2015 variant-classification criteria (Richards et al., Genet Med
2015;17:405-424), with ClinGen Sequence Variant Interpretation (SVI) modifications, so the
model quotes definitions instead of recalling them (it once described BS2 as BS1).

Each entry: the 2015 definition (paraphrased closely), its default strength, and the main
ClinGen SVI change where one applies. Gene-specific ClinGen VCEP specifications override
these general rules. Reviewed for the lab by a geneticist before clinical reliance.
"""
from __future__ import annotations

SOURCE = "ACMG/AMP 2015 (Richards et al., Genet Med 2015) + ClinGen SVI recommendations"

CRITERIA: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ pathogenic
    "PVS1": {
        "strength": "Very strong (pathogenic)",
        "definition": "Null variant (nonsense, frameshift, canonical +/-1 or 2 splice sites, "
                      "initiation codon, single or multi-exon deletion) in a gene where loss of "
                      "function (LOF) is a known mechanism of disease.",
        "cautions": "Be sure LOF is a disease mechanism for the gene; variants at the extreme "
                    "3' end may escape nonsense-mediated decay; splice variants may cause "
                    "in-frame exon skipping; consider alternative transcripts.",
        "svi": "ClinGen SVI PVS1 decision tree (Abou Tayoun et al. 2018) grades PVS1 as Very "
               "strong, Strong, Moderate or Supporting by variant type, NMD prediction, exon "
               "relevance and the proportion of protein removed.",
    },
    "PS1": {
        "strength": "Strong (pathogenic)",
        "definition": "Same amino acid change as a previously established pathogenic variant, "
                      "regardless of nucleotide change (e.g. Val->Leu caused by G>C or G>T at "
                      "the same codon).",
        "cautions": "Beware of changes that affect splicing rather than the amino acid. The "
                    "comparison variant must itself be established as pathogenic.",
        "svi": "ClinGen SVI has published guidance on PS1, including for splicing; check the "
               "current recommendation before applying it beyond missense changes.",
    },
    "PS2": {
        "strength": "Strong (pathogenic)",
        "definition": "De novo (both maternity and paternity confirmed) in a patient with the "
                      "disease and no family history.",
        "cautions": "Confirming parentage alone is not enough if the phenotype is not "
                    "consistent with the gene.",
        "svi": "ClinGen SVI de novo recommendation (PS2/PM6) uses a points system weighting "
               "confirmed vs assumed de novo and phenotype consistency, summed across cases.",
    },
    "PS3": {
        "strength": "Strong (pathogenic)",
        "definition": "Well-established in vitro or in vivo functional studies supportive of a "
                      "damaging effect on the gene or gene product.",
        "cautions": "Only validated, reproducible assays that reflect the disease mechanism.",
        "svi": "ClinGen SVI framework for PS3/BS3 (Brnich et al. 2019): strength depends on "
               "assay validation (number of pathogenic/benign controls).",
    },
    "PS4": {
        "strength": "Strong (pathogenic)",
        "definition": "The prevalence of the variant in affected individuals is significantly "
                      "increased compared with the prevalence in controls (e.g. relative risk "
                      "or odds ratio > 5.0 with a confidence interval not including 1.0).",
        "cautions": "For very rare variants, observation in several unrelated affected "
                    "patients may be counted at moderate or supporting level instead.",
        "svi": "",
    },
    "PM1": {
        "strength": "Moderate (pathogenic)",
        "definition": "Located in a mutational hot spot and/or critical and well-established "
                      "functional domain (e.g. active site of an enzyme) without benign "
                      "variation.",
        "cautions": "Needs gene/domain-specific evidence; generally applied to missense.",
        "svi": "",
    },
    "PM2": {
        "strength": "Moderate (pathogenic) in 2015 - see SVI change",
        "definition": "Absent from controls (or at extremely low frequency if recessive) in "
                      "population databases (Exome Sequencing Project, 1000 Genomes, ExAC; now "
                      "gnomAD).",
        "cautions": "Population data for insertions/deletions may be poorly called; check "
                    "coverage at the site.",
        "svi": "ClinGen SVI (2020): apply PM2 at SUPPORTING strength (PM2_Supporting). The "
               "SVI states PVS1 + PM2_Supporting may be combined to reach Likely pathogenic.",
    },
    "PM3": {
        "strength": "Moderate (pathogenic)",
        "definition": "For recessive disorders, detected in trans with a pathogenic variant.",
        "cautions": "Requires phase (parental testing) to confirm trans.",
        "svi": "ClinGen SVI PM3 recommendation uses points across probands, weighted by "
               "whether the other allele is pathogenic/likely pathogenic and phase confirmed.",
    },
    "PM4": {
        "strength": "Moderate (pathogenic)",
        "definition": "Protein length changes as a result of in-frame deletions/insertions in a "
                      "non-repeat region, or stop-loss variants.",
        "cautions": "Not for in-frame changes in repetitive regions without known function "
                    "(that is BP3).",
        "svi": "",
    },
    "PM5": {
        "strength": "Moderate (pathogenic)",
        "definition": "Novel missense change at an amino acid residue where a different "
                      "missense change determined to be pathogenic has been seen before (e.g. "
                      "Arg156His is pathogenic; now you observe Arg156Cys).",
        "cautions": "Beware of changes that affect splicing. The earlier variant must be "
                    "established as pathogenic.",
        "svi": "",
    },
    "PM6": {
        "strength": "Moderate (pathogenic)",
        "definition": "Assumed de novo, but without confirmation of paternity and maternity.",
        "cautions": "",
        "svi": "Covered by the ClinGen SVI de novo points system together with PS2.",
    },
    "PP1": {
        "strength": "Supporting (pathogenic)",
        "definition": "Cosegregation with disease in multiple affected family members in a gene "
                      "definitively known to cause the disease.",
        "cautions": "Can be upgraded to stronger evidence with more segregation data.",
        "svi": "",
    },
    "PP2": {
        "strength": "Supporting (pathogenic)",
        "definition": "Missense variant in a gene that has a low rate of benign missense "
                      "variation and in which missense variants are a common mechanism of "
                      "disease.",
        "cautions": "",
        "svi": "",
    },
    "PP3": {
        "strength": "Supporting (pathogenic)",
        "definition": "Multiple lines of computational evidence support a deleterious effect on "
                      "the gene or gene product (conservation, evolutionary, splicing impact, "
                      "etc.).",
        "cautions": "Computational tools often share data; do not count each tool separately.",
        "svi": "ClinGen SVI (Pejaver et al. 2022): use one calibrated predictor with calibrated "
               "score thresholds; PP3 can then reach Moderate or Strong for missense.",
    },
    "PP4": {
        "strength": "Supporting (pathogenic)",
        "definition": "Patient's phenotype or family history is highly specific for a disease "
                      "with a single genetic etiology.",
        "cautions": "Depends on the actual patient's phenotype. Does not apply to phenotypes "
                    "caused by many different genes.",
        "svi": "",
    },
    "PP5": {
        "strength": "Supporting (pathogenic)",
        "definition": "Reputable source recently reports variant as pathogenic, but the evidence "
                      "is not available to the laboratory to perform an independent evaluation.",
        "cautions": "",
        "svi": "ClinGen SVI (Biesecker & Harrison 2018): PP5 should NOT be used; evaluate the "
               "underlying evidence instead.",
    },
    # ------------------------------------------------------------------ benign
    "BA1": {
        "strength": "Stand-alone (benign)",
        "definition": "Allele frequency is > 5% in Exome Sequencing Project, 1000 Genomes or "
                      "ExAC (now gnomAD).",
        "cautions": "",
        "svi": "ClinGen SVI (Ghosh et al. 2018): > 0.05 in any general continental population "
               "dataset with at least 2,000 observed alleles, and the variant is not on the "
               "SVI BA1 exception list.",
    },
    "BS1": {
        "strength": "Strong (benign)",
        "definition": "Allele frequency is greater than expected for the disorder.",
        "cautions": "Needs a disease-specific threshold from prevalence, inheritance, genetic "
                    "and allelic heterogeneity and penetrance (e.g. maximum credible allele "
                    "frequency).",
        "svi": "",
    },
    "BS2": {
        "strength": "Strong (benign)",
        "definition": "Observed in a healthy adult individual for a recessive (homozygous), "
                      "dominant (heterozygous), or X-linked (hemizygous) disorder, with full "
                      "penetrance expected at an early age.",
        "cautions": "Not for late-onset or reduced-penetrance disorders. This is about "
                    "observing the genotype in an unaffected adult, not about frequency (that "
                    "is BS1).",
        "svi": "",
    },
    "BS3": {
        "strength": "Strong (benign)",
        "definition": "Well-established in vitro or in vivo functional studies show no damaging "
                      "effect on protein function or splicing.",
        "cautions": "",
        "svi": "ClinGen SVI framework for PS3/BS3 (Brnich et al. 2019).",
    },
    "BS4": {
        "strength": "Strong (benign)",
        "definition": "Lack of segregation in affected members of a family.",
        "cautions": "Beware phenocopies and non-genetic causes in families.",
        "svi": "",
    },
    "BP1": {
        "strength": "Supporting (benign)",
        "definition": "Missense variant in a gene for which primarily truncating variants are "
                      "known to cause disease.",
        "cautions": "",
        "svi": "",
    },
    "BP2": {
        "strength": "Supporting (benign)",
        "definition": "Observed in trans with a pathogenic variant for a fully penetrant "
                      "dominant gene/disorder, or observed in cis with a pathogenic variant in "
                      "any inheritance pattern.",
        "cautions": "",
        "svi": "",
    },
    "BP3": {
        "strength": "Supporting (benign)",
        "definition": "In-frame deletions/insertions in a repetitive region without a known "
                      "function.",
        "cautions": "",
        "svi": "",
    },
    "BP4": {
        "strength": "Supporting (benign)",
        "definition": "Multiple lines of computational evidence suggest no impact on the gene "
                      "or gene product (conservation, evolutionary, splicing impact, etc.).",
        "cautions": "",
        "svi": "ClinGen SVI (Pejaver et al. 2022): calibrated single-predictor thresholds; BP4 "
               "can reach Moderate or Strong.",
    },
    "BP5": {
        "strength": "Supporting (benign)",
        "definition": "Variant found in a case with an alternate molecular basis for disease.",
        "cautions": "Use with care for genes where multiple variants can contribute.",
        "svi": "",
    },
    "BP6": {
        "strength": "Supporting (benign)",
        "definition": "Reputable source recently reports variant as benign, but the evidence is "
                      "not available to the laboratory to perform an independent evaluation.",
        "cautions": "",
        "svi": "ClinGen SVI (Biesecker & Harrison 2018): BP6 should NOT be used.",
    },
    "BP7": {
        "strength": "Supporting (benign)",
        "definition": "A synonymous (silent) variant for which splicing prediction algorithms "
                      "predict no impact on the splice consensus sequence nor the creation of a "
                      "new splice site, AND the nucleotide is not highly conserved.",
        "cautions": "",
        "svi": "The ClinGen SVI Splicing Subgroup (2023) extended BP7 to some deep intronic "
               "variants; check the current recommendation.",
    },
}

COMBINING = {
    "pathogenic": [
        "1 Very strong AND >=1 Strong",
        "1 Very strong AND >=2 Moderate",
        "1 Very strong AND 1 Moderate AND 1 Supporting",
        "1 Very strong AND >=2 Supporting",
        ">=2 Strong",
        "1 Strong AND >=3 Moderate",
        "1 Strong AND 2 Moderate AND >=2 Supporting",
        "1 Strong AND 1 Moderate AND >=4 Supporting",
    ],
    "likely_pathogenic": [
        "1 Very strong AND 1 Moderate",
        "1 Strong AND 1-2 Moderate",
        "1 Strong AND >=2 Supporting",
        ">=3 Moderate",
        "2 Moderate AND >=2 Supporting",
        "1 Moderate AND >=4 Supporting",
        "(ClinGen SVI 2020) PVS1 AND PM2_Supporting",
    ],
    "benign": ["1 Stand-alone (BA1)", ">=2 Strong benign"],
    "likely_benign": ["1 Strong benign AND 1 Supporting benign", ">=2 Supporting benign"],
    "otherwise": "Uncertain significance (VUS), including when pathogenic and benign "
                 "criteria contradict each other.",
    "points_alternative": "Bayesian points (Tavtigian et al. 2020): Supporting 1, Moderate 2, "
                          "Strong 4, Very strong 8 (benign criteria negative). Total >=10 "
                          "Pathogenic, 6-9 Likely pathogenic, 0-5 VUS, -1 to -6 Likely benign, "
                          "<=-7 Benign.",
}


def acmg_criterion(code: str) -> dict:
    """One or several criteria ("BS2" or "PVS1, PM4, BS1"), "all" (one-line list), or
    "combining" (classification rules). Several in one call saves the model a round trip
    per criterion, which matters under Groq's per-minute token limit."""
    parts = [p for p in str(code or "").replace(";", ",").replace(" ", ",").split(",") if p]
    if len(parts) > 1:
        # Compact form (strength, definition, SVI change) and at most 8 codes: Groq's free
        # tier allows ~8,000 tokens/minute and a long list once used all of it.
        out = {}
        for p in parts[:8]:
            d = _one(p)
            out[p.upper()] = {k: d[k] for k in ("strength", "definition", "svi", "error")
                              if d.get(k)}
        return {"criteria": out, "source": SOURCE,
                **({"skipped": parts[8:]} if len(parts) > 8 else {}),
                "note": "Gene-specific ClinGen VCEP specifications, where they exist, override "
                        "these general rules."}
    return _one(code)


def _one(code: str) -> dict:
    c = (code or "").strip().upper().replace(" ", "").split("_")[0]
    if c in ("COMBINING", "COMBINE", "RULES", "CLASSIFICATION"):
        return {"combining_rules": COMBINING, "source": SOURCE}
    if c in ("ALL", "LIST", ""):
        return {"criteria": {k: f"{v['strength']}: {v['definition']}" for k, v in CRITERIA.items()},
                "source": SOURCE}
    if c not in CRITERIA:
        return {"error": f"unknown criterion {code!r}. Codes: {', '.join(CRITERIA)}; or "
                         "'combining' for the classification rules."}
    return {"code": c, **{k: v for k, v in CRITERIA[c].items() if v}, "source": SOURCE}
