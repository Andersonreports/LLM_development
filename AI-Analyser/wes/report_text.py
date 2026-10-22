"""Fixed wording of the laboratory report. Taken from the lab's "no pathogenic
variant" report template - edit here to change what every report says."""
from __future__ import annotations

LAB_METHODOLOGY_UPSTREAM = (
    "DNA was used to perform whole exome using whole exome capture kit. The targeted libraries were sequenced to a "
    "targeted depth of 80X to 100X using SURFSeq 5000 sequencing platform. This kit has deep exonic coverage of all "
    "the coding regions including the difficult to cover regions. The sequences obtained are aligned to human "
    "reference genome (GRCh38.p13) using Sentieon aligner and analyzed using Sentieon for removing duplicates, "
    "recalibration and re-alignment of indels. Sentieon DNAscope has been used to call the variants."
)

# Describes what THIS pipeline does after variant calling. Keep in sync with annotate.py / acmg.py.
PIPELINE_METHODOLOGY = (
    "Called variants were normalised, restricted to the exome capture targets and annotated with Ensembl VEP "
    "(release 116, MANE Select transcripts) using gnomAD (v4.1), ClinVar, OMIM/HPO gene-disease associations, "
    "dbSNP and NCBI RefSeq genes. In-silico predictions were obtained from SIFT, PolyPhen-2, AlphaMissense and, where "
    "available, REVEL and SpliceAI. Variants were filtered by quality, population frequency, predicted consequence "
    "and mode of inheritance, prioritised against the patient's HPO-coded phenotype, and classified according to the "
    "ACMG/AMP guidelines [1]. Only non-synonymous and splice site variants found in the coding regions were used for "
    "clinical interpretation. Silent variations that do not result in any change in amino acid in the coding region "
    "are not reported. The gene list used in the analysis is given in Appendix 1."
)

CLASSIFICATION_INTRO = (
    "Genetic test results are reported based on the recommendations of American College of Medical Genetics [1], "
    "as described below:"
)

CLASSIFICATION_TABLE = [
    ("Pathogenic",
     "A disease-causing variation in a gene which can explain the patients' symptoms has been detected. This usually "
     "means that a suspected disorder for which testing had been requested has been confirmed"),
    ("Likely Pathogenic",
     "A variant which is very likely to contribute to the development of disease however, the scientific evidence is "
     "currently insufficient to prove this conclusively. Additional evidence is expected to confirm this assertion of "
     "pathogenicity."),
    ("Variant of Uncertain Significance",
     "A variant has been detected, but it is difficult to classify it as either pathogenic (disease causing) or benign "
     "(non-disease causing) based on current available scientific evidence. Further testing of the patient or family "
     "members as recommended by your clinician may be needed. It is probable that their significance can be assessed "
     "only with time, subject to availability of scientific evidence."),
]

DISCLAIMER = [
    "The classification of variants of unknown significance can change over time. {lab} cannot be held responsible for it.",
    "Intronic variants, UTR and Promoter region variants are not assessed using this assay.",
    "This assay has a sensitivity of 70-75% in detecting large deletions/duplications of more than 10 base pairs or "
    "copy number variations (CNV). However, it is important to note that any CNVs detected must be confirmed using an "
    "alternate method.",
    "Certain genes may not be covered completely, and few mutations could be missed. Variants not detected by this "
    "assay may impact the phenotype.",
    "The variations have not been validated by Sanger sequencing.",
    "The above findings and result interpretation was done based on the clinical indication provided at the time of "
    "reporting.",
    "It is also possible that a pathogenic variant is present in a gene that was not selected for analysis and/or "
    "interpretation in cases where insufficient phenotypic information is available.",
    "Genes with pseudogenes, paralog genes and genes with low complexity may have decreased sensitivity and "
    "specificity of variant detection and interpretation due to inability of the data and analysis tools to "
    "unambiguously determine the origin of the sequence data in such regions.",
    "The variants of uncertain significance and variations with high minor allele frequencies which are likely to be "
    "benign will be given upon request.",
    "In addition, this test does not cover detection of such structural variants in the genes <i>SMN1</i>, "
    "<i>SMN2</i>, <i>HBA1</i>, <i>HBA2</i>, <i>PMS2</i>, <i>GBA</i>, and <i>CYP21A2</i> which have very high "
    "homology. Moreover, the ORF15 transcript isoform (RefSeq id: NM_001034853.1) of the <i>RPGR</i> gene associated "
    "with retinitis pigmentosa is not sufficiently covered in the test.",
    "Incidental or secondary findings that meet the ACMG guidelines can be given upon request [3].",
]

REFERENCES = [
    "Richards, S, et al. Standards and Guidelines for the Interpretation of Sequence Variants: A Joint Consensus "
    "Recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular "
    "Pathology. Genetics in medicine: official journal of the American College of Medical Genetics. 17.5 (2015): "
    "405-424.",
    "Amberger J, Bocchini CA, Scott AF, Hamosh A. McKusick's Online Mendelian Inheritance in Man (OMIM). Nucleic "
    "Acids Res. 2009 Jan;37(Database issue):D793-6. doi: 10.1093/nar/gkn665. Epub 2008 Oct 8.",
    "Lee, Kristy et al. “ACMG SF v3.3 list for reporting of secondary findings in clinical exome and genome "
    "sequencing: A policy statement of the American College of Medical Genetics and Genomics (ACMG).” Genetics "
    "in medicine : official journal of the American College of Medical Genetics vol. 27,8 (2025): 101454. "
    "doi:10.1016/j.gim.2025.101454.",
]

# Signatories printed under "This report has been reviewed and approved by"
SIGNATORIES = [
    ("Sivasankar.S, Ph.D", "Molecular Biologist"),
    ("Muthukumaran. S, Ph.D", "Clinical Bioinformatician"),
    ("Dr Robert Patrick Selvam", "MBBS, MD (Pathology), DM (Medical Genetics)"),
    ("Sachin. D.Honguntikar, Ph.D", "Molecular Geneticist"),
    ("Dr. G. Suriyakumar", "Director"),
]

NEGATIVE_RESULT = "No pathogenic or likely pathogenic variant(s) causative of the reported phenotype was detected"
NEGATIVE_FOOTNOTE = [
    "*Genetic test results are based on the recommendation of American college of Medical Genetics [1].",
    "No other variant that warrants to be reported for the given clinical indication was identified.",
]
CNV_NEGATIVE = "No significant Copy Number Variations (CNV) related to phenotype was detected."
RECOMMENDATIONS_NEGATIVE = [
    "Alternative test is strongly recommended to rule out the deletion/duplication.",
    "Genetic counselling is advised.",
]
