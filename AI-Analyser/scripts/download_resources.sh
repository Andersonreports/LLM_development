#!/usr/bin/env bash
# Downloads all offline annotation resources for GRCh38 into ./resources.
# Total ~30 GB (VEP cache ~26 GB). Safe to re-run: completed files are skipped.
#
#   ./scripts/download_resources.sh            # everything
#   ./scripts/download_resources.sh small      # everything except VEP cache + FASTA
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RES="$ROOT/resources"
BIN="${WES_TOOLS_BIN:-$ROOT/tools/mamba/envs/wes/bin}"
export PATH="$BIN:$PATH"
MODE="${1:-all}"
mkdir -p "$RES"/{vep_cache,fasta,clinvar,hpo,gene,plugins}

fetch() {  # fetch URL DEST
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "[skip] $dest"; return; fi
  echo "[get ] $url"
  if command -v aria2c >/dev/null; then
    # Ensembl throttles per connection; parallel ranges are ~25x faster
    aria2c -x 16 -s 16 -k 20M --continue=true --max-tries=10 --retry-wait=10 \
      --summary-interval=60 --console-log-level=warn \
      -d "$(dirname "$dest")" -o "$(basename "$dest").part" "$url"
  else
    curl -fL --retry 5 --retry-delay 10 -C - -o "$dest.part" "$url"
  fi
  mv "$dest.part" "$dest"
}

# --- Phenotype / gene-disease (HPO, free) --------------------------------
HPO=https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download
fetch "$HPO/hp.obo"                 "$RES/hpo/hp.obo"
fetch "$HPO/genes_to_phenotype.txt" "$RES/hpo/genes_to_phenotype.txt"
fetch "$HPO/genes_to_disease.txt"   "$RES/hpo/genes_to_disease.txt"
fetch "$HPO/phenotype.hpoa"         "$RES/hpo/phenotype.hpoa"

# --- Gene constraint and dosage sensitivity -------------------------------
fetch https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv \
      "$RES/gene/gnomad.v4.1.constraint_metrics.tsv"
fetch https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv \
      "$RES/gene/ClinGen_gene_curation_list_GRCh38.tsv"

# --- ClinVar (weekly release, no 'chr' prefix) ----------------------------
fetch https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz     "$RES/clinvar/clinvar.vcf.gz"
fetch https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi "$RES/clinvar/clinvar.vcf.gz.tbi"

# --- AlphaMissense (VEP plugin data, CC BY-NC-SA 4.0) ---------------------
fetch https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz "$RES/plugins/AlphaMissense_hg38.tsv.gz"
if [ ! -s "$RES/plugins/AlphaMissense_hg38.tsv.gz.tbi" ]; then
  "$BIN/tabix" -s 1 -b 2 -e 2 -f -S 1 "$RES/plugins/AlphaMissense_hg38.tsv.gz"
fi

if [ "$MODE" = "small" ]; then echo "[done] small resources"; exit 0; fi

# --- VEP cache matching installed VEP release ----------------------------
VEP_VER="$("$BIN/vep" --help 2>/dev/null | sed -n 's/.*ensembl-vep *: *\([0-9]\+\).*/\1/p' | head -1)"
if [ -z "$VEP_VER" ]; then echo "Cannot detect VEP version; run scripts/setup_tools.sh first" >&2; exit 1; fi
echo "[info] VEP release $VEP_VER"
if [ ! -d "$RES/vep_cache/homo_sapiens/${VEP_VER}_GRCh38" ]; then
  TAR="$RES/vep_cache/homo_sapiens_vep_${VEP_VER}_GRCh38.tar.gz"
  fetch "https://ftp.ensembl.org/pub/release-${VEP_VER}/variation/indexed_vep_cache/homo_sapiens_vep_${VEP_VER}_GRCh38.tar.gz" "$TAR"
  tar -xzf "$TAR" -C "$RES/vep_cache" && rm -f "$TAR"
fi

# --- Reference FASTA (Ensembl naming: 1..22,X,Y,MT), bgzipped + indexed ----
FA="$RES/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
if [ ! -s "$FA.fai" ]; then
  fetch "https://ftp.ensembl.org/pub/release-${VEP_VER}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz" "$FA.orig"
  gunzip -c "$FA.orig" | "$BIN/bgzip" -@ 8 -c > "$FA"
  "$BIN/samtools" faidx "$FA"
  rm -f "$FA.orig"
fi

echo "[done] all resources in $RES"
