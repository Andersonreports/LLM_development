#!/bin/bash
# Run ONE case through ADVAT headless, on the LOCAL model (qwen3.5:4b) - no web UI, no Claude.
# Produces <OUT>/<ACCESSION>_report.html, _report.pdf and _result.json.
#
# Usage (inside WSL, or via `wsl -d Ubuntu -u root -- bash <this> ...`):
#   run_case.sh <vcf> <clinical_history_txt_or_text> <out_dir> <accession> [age] [sex] [indication]
#
# Example:
#   run_case.sh /mnt/c/.../case.vcf /mnt/c/.../history.txt /mnt/c/.../out CASE42 "5 years" M "Epilepsy"
set -u
[ $# -ge 4 ] || { echo "Usage: $0 <vcf> <clinical> <out_dir> <accession> [age] [sex] [indication]"; exit 1; }

REPO=/mnt/c/Users/Anderson/Desktop/Projects/LLM_development/AI-Analyser
export WES_RESOURCES_DIR=/root/wes/resources
export WES_TOOLS_BIN=/root/wes/tools/mamba/envs/wes/bin
export WES_AI_BACKEND=local            # our local model, not Claude
export WES_OLLAMA_MODEL=qwen3.5:4b     # the bake-off winner
export WES_OLLAMA_THINK=false
export PYTHONPATH="$REPO"
cd "$REPO" || exit 1

/root/wes/venv/bin/python scripts/run_case_headless.py \
  --vcf "$1" --clinical "$2" --out "$3" --accession "$4" \
  --age "${5:-}" --sex "${6:-}" --indication "${7:-}"
