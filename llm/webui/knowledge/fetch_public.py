"""Download small public reference files the chat reads directly (no database rebuild).

    python -m webui.knowledge.fetch_public          # from llm/

Currently: ClinGen gene-disease validity curations (search.clinicalgenome.org), saved to
llm/data/clingen_gene_validity.csv (git-ignored). Re-run to refresh; the file records its
own creation date, which the chat cites.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

from ..config import ROOT

DATA = ROOT / "data"
FILES = {
    "clingen_gene_validity.csv": "https://search.clinicalgenome.org/kb/gene-validity/download",
}


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, url in FILES.items():
        dest = DATA / name
        try:
            r = httpx.get(url, timeout=120, follow_redirects=True,
                          headers={"User-Agent": "advat-llm-chat/1.0"})
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"FAILED {name}: {e}", file=sys.stderr)
            ok = False
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)                      # never leave a half-written file in place
        print(f"saved {dest} ({len(r.content) // 1024} KB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
