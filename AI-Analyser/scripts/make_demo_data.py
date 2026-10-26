"""Build a self-contained demo case so the pipeline can be exercised without
bcftools, VEP or the 30 GB resource bundle.

    python scripts/make_demo_data.py

Writes to demo/:
  resources/          a miniature knowledge base (HPO ontology, gene-disease,
                      gene-phenotype, constraint, ClinGen dosage)
  demo_proband.vcf    a VEP-annotated single-sample VCF - because it already carries
                      CSQ, prepare_case_vcf() skips preprocessing and annotation entirely
  README.txt          how to run it

Everything is invented. No patient data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from conftest import CLINGEN, CONSTRAINT, G2D, G2P, HP_OBO, HPOA, csq  # noqa: E402

OUT = ROOT / "demo"
RES = OUT / "resources"

# (chrom, pos, ref, alt, CSQ, genotype) - a realistic small candidate list
ROWS = [
    # the answer: de novo nonsense in SCN1A, matches the clinical history
    ("2", 166848646, "C", "T",
     csq("T", "stop_gained", "HIGH", "SCN1A", exon="5/26", hgvsc="c.664C>T", hgvsp="p.Arg222Ter"),
     "0/1"),
    # a rare missense in an unrelated gene - should be ranked below, not reported
    ("5", 140100000, "A", "G",
     csq("G", "missense_variant", "MODERATE", "OTHER1", hgvsc="c.310A>G", hgvsp="p.Lys104Glu",
         af="0.00008", am="0.31"),
     "0/1"),
    # a common variant - must be filtered out before ranking
    ("5", 140100500, "T", "C",
     csq("C", "missense_variant", "MODERATE", "OTHER1", hgvsc="c.800T>C", af="0.21"),
     "1/1"),
    # a carrier finding: one pathogenic PAH allele, heterozygous
    ("12", 102852882, "G", "A",
     csq("A", "missense_variant", "MODERATE", "PAH", exon="12/13", hgvsc="c.1222C>T",
         hgvsp="p.Arg408Trp", af="0.00008", clnsig="Pathogenic",
         clnrev="criteria_provided,_multiple_submitters,_no_conflicts", am="0.98"),
     "0/1"),
    # a secondary finding (ACMG SF gene), reported only if consent is given
    ("13", 32339000, "C", "T",
     csq("T", "stop_gained", "HIGH", "BRCA2", exon="11/27", hgvsc="c.5946C>T", hgvsp="p.Ser1982Ter"),
     "0/1"),
    # synonymous, no splice effect - should be dropped
    ("12", 102853000, "G", "A",
     csq("A", "synonymous_variant", "LOW", "PAH", hgvsc="c.1233C>T"),
     "0/1"),
]

VCF_HEADER = """##fileformat=VCFv4.2
##reference=GRCh38
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. \
Format: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|\
MANE_SELECT|CANONICAL|VARIANT_CLASS|gnomADe_AF|gnomADg_AF|MAX_AF|MAX_AF_POPS|\
ClinVar_CLNSIG|ClinVar_CLNREVSTAT|ClinVar_CLNDN|am_pathogenicity|am_class|\
Existing_variation|SIFT|PolyPhen">
"""


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    for sub, name, text in [
        ("hpo", "hp.obo", HP_OBO), ("hpo", "genes_to_phenotype.txt", G2P),
        ("hpo", "genes_to_disease.txt", G2D), ("hpo", "phenotype.hpoa", HPOA),
        ("gene", "gnomad.v4.1.constraint_metrics.tsv", CONSTRAINT),
        ("gene", "ClinGen_gene_curation_list_GRCh38.tsv", CLINGEN),
    ]:
        (RES / sub).mkdir(parents=True, exist_ok=True)
        (RES / sub / name).write_text(text, encoding="utf-8")

    vcf = OUT / "demo_proband.vcf"
    lines = [VCF_HEADER.rstrip("\n"), "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tDEMO01"]
    for chrom, pos, ref, alt, info, gt in ROWS:
        depth = 48
        ad = f"{depth // 2},{depth // 2}" if gt == "0/1" else f"0,{depth}"
        lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t900\tPASS\tCSQ={info}\tGT:DP:GQ:AD\t"
                     f"{gt}:{depth}:99:{ad}")
    vcf.write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "README.txt").write_text(
        "Demo data (invented - no patient data).\n\n"
        "Start the server pointed at the miniature knowledge base:\n"
        f"  $env:WES_RESOURCES_DIR='{RES}'\n"
        "  .venv\\Scripts\\uvicorn wes.web.app:app --port 8000\n\n"
        "Then create a single case at http://localhost:8000, upload demo_proband.vcf,\n"
        "and use a clinical history such as:\n"
        "  'Three-year-old girl with febrile seizures from six months of age, now with\n"
        "   prolonged generalised tonic-clonic seizures and global developmental delay.'\n"
        "Tick both consent boxes to see secondary and carrier findings.\n\n"
        "Expected: POSITIVE, SCN1A c.664C>T as the primary finding, PAH as a carrier\n"
        "finding, BRCA2 as a secondary finding, and the common variant filtered out.\n",
        encoding="utf-8")

    print(f"knowledge base : {RES}")
    print(f"demo VCF       : {vcf}  ({len(ROWS)} variants, VEP-annotated)")
    print(f"instructions   : {OUT / 'README.txt'}")


if __name__ == "__main__":
    main()
