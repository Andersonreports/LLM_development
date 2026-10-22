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

# VEP must be pinned: left unpinned, the solver can pick an ancient release (e.g. 88)
# that is broken and does not match the VEP cache this pipeline is written for
VEP_RELEASE=116
if ! "$MAMBA_ROOT_PREFIX/envs/wes/bin/vep" --help 2>/dev/null | grep -q "ensembl-vep *: *$VEP_RELEASE"; then
  echo "[setup] creating bioconda env 'wes' (ensembl-vep $VEP_RELEASE, bcftools, htslib)"
  "$TOOLS/bin/micromamba" env remove -y -n wes >/dev/null 2>&1 || true
  "$TOOLS/bin/micromamba" create -y -n wes -c conda-forge -c bioconda \
    "ensembl-vep=$VEP_RELEASE" bcftools htslib samtools aria2
fi

BIN="$MAMBA_ROOT_PREFIX/envs/wes/bin"
export PATH="$BIN:$PATH"
"$BIN/vep" --help 2>/dev/null | grep -m1 -i "ensembl-vep" || true
"$BIN/bcftools" --version | head -1
echo "[setup] tools ready in $BIN"
echo "WES_TOOLS_BIN=$BIN"
