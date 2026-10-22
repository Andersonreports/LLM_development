"""Test fixtures: a tiny knowledge base and synthetic VEP-annotated VCFs.

Environment is set before any ``wes`` import so Settings picks it up.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="wes-test-"))
RES = TMP / "resources"
os.environ["WES_DATA_DIR"] = str(TMP / "data")
os.environ["WES_RESOURCES_DIR"] = str(RES)
os.environ["WES_AI_ENABLED"] = "0"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402

# ------------------------------------------------------------------ knowledge base
HP_OBO = """format-version: 1.2

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0000118
name: Phenotypic abnormality
is_a: HP:0000001 ! All

[Term]
id: HP:0000707
name: Abnormality of the nervous system
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
synonym: "Seizures" EXACT []
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0002373
name: Febrile seizure (within the age range of 3 months to 6 years)
is_a: HP:0001250 ! Seizure

[Term]
id: HP:0001263
name: Global developmental delay
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0001939
name: Abnormality of metabolism/homeostasis
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0004923
name: Hyperphenylalaninemia
is_a: HP:0001939 ! Abnormality of metabolism/homeostasis

[Term]
id: HP:0003198
name: Myopathy
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0006528
name: Chronic lung disease
is_a: HP:0000118 ! Phenotypic abnormality
"""

G2P = "ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id\n" + "\n".join([
    "6323\tSCN1A\tHP:0001250\tSeizure\t-\tOMIM:607208",
    "6323\tSCN1A\tHP:0002373\tFebrile seizure\t-\tOMIM:607208",
    "6323\tSCN1A\tHP:0001263\tGlobal developmental delay\t-\tOMIM:607208",
    "5053\tPAH\tHP:0004923\tHyperphenylalaninemia\t-\tOMIM:261600",
    "5053\tPAH\tHP:0001263\tGlobal developmental delay\t-\tOMIM:261600",
    "5053\tPAH\tHP:0001250\tSeizure\t-\tOMIM:261600",
    "1756\tDMD\tHP:0003198\tMyopathy\t-\tOMIM:310200",
    "1080\tCFTR\tHP:0006528\tChronic lung disease\t-\tOMIM:219700",
    "675\tBRCA2\tHP:0000118\tPhenotypic abnormality\t-\tOMIM:612555",
    "9999\tOTHER1\tHP:0003198\tMyopathy\t-\tOMIM:100000",
]) + "\n"

G2D = "ncbi_gene_id\tgene_symbol\tassociation_type\tdisease_id\tsource\n" + "\n".join([
    "6323\tSCN1A\tMENDELIAN\tOMIM:607208\tx",
    "5053\tPAH\tMENDELIAN\tOMIM:261600\tx",
    "1756\tDMD\tMENDELIAN\tOMIM:310200\tx",
    "1080\tCFTR\tMENDELIAN\tOMIM:219700\tx",
    "675\tBRCA2\tMENDELIAN\tOMIM:612555\tx",
    "9999\tOTHER1\tMENDELIAN\tOMIM:100000\tx",
]) + "\n"

HPOA = "#description: test\ndatabase_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n" + "\n".join([
    "OMIM:607208\tDravet syndrome\t\tHP:0000006\tx\tTAS\t\t\t\t\tI\tx",
    "OMIM:607208\tDravet syndrome\t\tHP:0001250\tx\tTAS\t\t\t\t\tP\tx",
    "OMIM:261600\tPhenylketonuria\t\tHP:0000007\tx\tTAS\t\t\t\t\tI\tx",
    "OMIM:310200\tDuchenne muscular dystrophy\t\tHP:0001419\tx\tTAS\t\t\t\t\tI\tx",
    "OMIM:219700\tCystic fibrosis\t\tHP:0000007\tx\tTAS\t\t\t\t\tI\tx",
    "OMIM:612555\tBreast-ovarian cancer, familial, 2\t\tHP:0000006\tx\tTAS\t\t\t\t\tI\tx",
    "OMIM:100000\tOther myopathy\t\tHP:0000006\tx\tTAS\t\t\t\t\tI\tx",
]) + "\n"

CONSTRAINT = "gene\ttranscript\tmane_select\tlof.oe_ci.upper\tlof.pLI\tmis.z_score\n" + "\n".join([
    "SCN1A\tENST1\ttrue\t0.1\t1.0\t5.5",
    "PAH\tENST2\ttrue\t0.9\t0.0\t1.0",
    "DMD\tENST3\ttrue\t0.2\t1.0\t2.0",
    "CFTR\tENST4\ttrue\t0.8\t0.0\t1.0",
    "BRCA2\tENST5\ttrue\t0.3\t1.0\t0.5",
    "OTHER1\tENST6\ttrue\t1.2\t0.0\t0.0",
]) + "\n"

CLINGEN = "#ClinGen\n#Gene Symbol\tGene ID\tcytoBand\tGenomic Location\tHaploinsufficiency Score\tHaploinsufficiency Description\tTriplosensitivity Score\n" + \
    "SCN1A\t6323\t2q24.3\tx\t3\tSufficient\t0\nBRCA2\t675\t13q13.1\tx\t3\tSufficient\t0\n"


def _write_kb() -> None:
    (RES / "hpo").mkdir(parents=True, exist_ok=True)
    (RES / "gene").mkdir(parents=True, exist_ok=True)
    (RES / "hpo" / "hp.obo").write_text(HP_OBO)
    (RES / "hpo" / "genes_to_phenotype.txt").write_text(G2P)
    (RES / "hpo" / "genes_to_disease.txt").write_text(G2D)
    (RES / "hpo" / "phenotype.hpoa").write_text(HPOA)
    (RES / "gene" / "gnomad.v4.1.constraint_metrics.tsv").write_text(CONSTRAINT)
    (RES / "gene" / "ClinGen_gene_curation_list_GRCh38.tsv").write_text(CLINGEN)


_write_kb()

# ------------------------------------------------------------------ VCF builder
CSQ_FIELDS = ("Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|"
              "Existing_variation|CANONICAL|MANE_SELECT|gnomADe_AF|gnomADg_AF|MAX_AF|MAX_AF_POPS|ClinVar|"
              "ClinVar_CLNSIG|ClinVar_CLNREVSTAT|ClinVar_CLNDN|am_class|am_pathogenicity|SIFT|PolyPhen|VARIANT_CLASS")


def csq(alt, cons, impact, gene, exon="5/20", hgvsc="", hgvsp="", af="", clnsig="", clnrev="", am=""):
    vals = {
        "Allele": alt, "Consequence": cons, "IMPACT": impact, "SYMBOL": gene, "Gene": f"ENSG_{gene}",
        "Feature_type": "Transcript", "Feature": f"ENST_{gene}", "BIOTYPE": "protein_coding", "EXON": exon,
        "INTRON": "", "HGVSc": f"ENST_{gene}:{hgvsc}" if hgvsc else "", "HGVSp": f"ENSP_{gene}:{hgvsp}" if hgvsp else "",
        "Existing_variation": "", "CANONICAL": "YES", "MANE_SELECT": f"NM_{gene}.1", "gnomADe_AF": af,
        "gnomADg_AF": "", "MAX_AF": af, "MAX_AF_POPS": "", "ClinVar": "1" if clnsig else "",
        "ClinVar_CLNSIG": clnsig, "ClinVar_CLNREVSTAT": clnrev, "ClinVar_CLNDN": "", "am_class": "",
        "am_pathogenicity": am, "SIFT": "", "PolyPhen": "", "VARIANT_CLASS": "SNV",
    }
    # VEP encodes commas inside CSQ values as "&"
    return "|".join(vals[f].replace(",", "&") for f in CSQ_FIELDS.split("|"))


def write_vcf(path: Path, samples: list[str], rows: list[tuple]) -> Path:
    """rows: (chrom, pos, ref, alt, csq_string, [GT per sample])"""
    lines = [
        "##fileformat=VCFv4.2",
        "##reference=GRCh38",
        "##contig=<ID=chr1,length=248956422>",
        '##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. Format: ' + CSQ_FIELDS + '">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">',
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="GQ">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="AD">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for chrom, pos, ref, alt, c, gts in rows:
        sample_cols = []
        for gt in gts:
            ad = {"0/0": "40,0", "0/1": "20,20", "1/1": "0,40", "./.": ".", "1": "0,40"}.get(gt, "20,20")
            sample_cols.append(f"{gt}:40:99:{ad}")
        lines.append(f"chr{chrom}\t{pos}\t.\t{ref}\t{alt}\t500\tPASS\tCSQ={c}\tGT:DP:GQ:AD\t" + "\t".join(sample_cols))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def tmpdir_path(tmp_path):
    return tmp_path
