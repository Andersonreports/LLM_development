#!/usr/bin/env bash
# Installs micromamba (user-space, no sudo) and a bioconda env with
# Ensembl VEP, bcftools and htslib into ./tools.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
mkdir -p "$TOOLS"
export MAMBA_ROOT_PREFIX="$TOOLS/mamba"

if [ ! -x "$TOOLS/bin/micromamba" ]; then
  echo "[setup] downloading micromamba"
  mkdir -p "$TOOLS/bin"
  curl -fLs -o "$TOOLS/bin/micromamba" \
    https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x "$TOOLS/bin/micromamba"
fi

if [ ! -x "$MAMBA_ROOT_PREFIX/envs/wes/bin/vep" ]; then
  echo "[setup] creating bioconda env 'wes' (ensembl-vep, bcftools, htslib)"
  "$TOOLS/bin/micromamba" create -y -n wes -c conda-forge -c bioconda \
    ensembl-vep bcftools htslib samtools aria2
fi

BIN="$MAMBA_ROOT_PREFIX/envs/wes/bin"
export PATH="$BIN:$PATH"
"$BIN/vep" --help 2>/dev/null | grep -m1 -i "ensembl-vep" || true
"$BIN/bcftools" --version | head -1
echo "[setup] tools ready in $BIN"
echo "WES_TOOLS_BIN=$BIN"
