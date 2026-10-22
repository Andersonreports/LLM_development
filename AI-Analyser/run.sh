#!/usr/bin/env bash
# Start the web app on http://0.0.0.0:8000
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/uvicorn wes.web.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
