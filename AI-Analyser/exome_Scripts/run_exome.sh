#!/bin/bash
# Single-sample exome analysis, fully local (no web calls).
# Usage: run_exome.sh <sample.vcf.gz> <output_dir>
#   <sample.vcf.gz>  DNAscope (or GATK) VCF, GRCh38, chr-prefixed, bgzipped
#   <output_dir>     created if missing; all results go here
# Needs: conda env "exome", VEP 116 GRCh38 cache in ~/.vep
set -euo pipefail
[ $# -eq 2 ] || { echo "Usage: $0 <sample.vcf.gz> <output_dir>"; exit 1; }
V=$(readlink -f "$1"); OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)          # exome_Scripts folder
REF=$HERE/reference
source ~/anaconda3/etc/profile.d/conda.sh; conda activate exome
mkdir -p "$OUT"; cd "$OUT"
rm -rf acmg_local; cp -r "$HERE/acmg_local" .
cp "$HERE/helpers/parse_vep.py" "$HERE/helpers/build_local_evidence.py" .
[ -f "$V.tbi" ] || [ -f "$V.csi" ] || { cp "$V" input.vcf.gz; bcftools index -t input.vcf.gz; V=$PWD/input.vcf.gz; }

echo "[1] filter: PASS, exome targets, real variants only"
bcftools view -f PASS,. -i 'FILTER="PASS" || QUAL>=30' -T "$REF/targets.bed" "$V" -Oz -o pass_on_target.vcf.gz; bcftools index -t pass_on_target.vcf.gz
bcftools view -e 'GT="ref" || GT="mis" || ALT="."' pass_on_target.vcf.gz -Oz -o variants_for_annotation.vcf.gz; bcftools index -t variants_for_annotation.vcf.gz
echo "counts: input $(bcftools view -H "$V" | wc -l)  on-target $(bcftools view -H pass_on_target.vcf.gz | wc -l)  variants $(bcftools view -H variants_for_annotation.vcf.gz | wc -l)"

echo "[2] ClinVar (local file)"
bcftools annotate -a "$REF/clinvar_chr.vcf.gz" -c INFO/CLNSIG,INFO/CLNREVSTAT,INFO/CLNDN,INFO/GENEINFO,INFO/MC \
  variants_for_annotation.vcf.gz -Oz -o annotated_clinvar.vcf.gz; bcftools index -t annotated_clinvar.vcf.gz

echo "[3] VEP (offline cache)"
vep --offline --cache --dir_cache ~/.vep --assembly GRCh38 --cache_version 116 --format vcf \
  -i variants_for_annotation.vcf.gz --vcf -o vep_annotated.vcf --force_overwrite --symbol --canonical --biotype \
  --sift b --polyphen b --af_gnomade --af_gnomadg --max_af --check_existing --fork 4 --no_stats

echo "[4] tables + local evidence"
python parse_vep.py > /dev/null
python build_local_evidence.py

echo "[5] ACMG/AMP classification (offline)"
python acmg_local/clinical_variant_reporter.py --input variants_for_annotation.vcf.gz --output acmg_out 2>&1 | tail -9

echo "[6] rare loss-of-function count"
RARE=$HERE/clawbio_skills/rare-high-impact-variants/rare_high_impact_variants.py
[ -f "$RARE" ] && python "$RARE" --input for_rare_skill.vcf --output rare_out | tail -2 || echo "  (skipped: ClawBio rare-variant skill not installed)"

echo "[7] ClinVar table for review"
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/GENEINFO\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\t%INFO/CLNDN\t[%GT\t%DP\t%AD\t%GQ]\n' annotated_clinvar.vcf.gz > clinvar_table.tsv
echo PIPELINE_DONE "$OUT"
