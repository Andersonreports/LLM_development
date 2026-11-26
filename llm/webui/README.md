# ADVAT LLM &middot; chat UI

A small web chat interface for this project, with a **backend switch** you flip in the browser:

| Backend | What it is | Data | Quality |
|---------|------------|------|---------|
| **tinygpt** | our own from-scratch model (`../tinygpt`), loaded once and kept warm in a background thread | stays on this machine | weak — it's a character-level *completion* model (it continues your text), not a chat assistant |
| **Groq** | Groq Cloud via its OpenAI-compatible API, using your `GROQ_API_KEY` | **sent to Groq's servers** | genuinely capable chat models (Llama, etc.), fast |

## About Groq

Groq does **not** run our own model. It serves a fixed catalogue of open-weight models very
fast. So "using Groq" means the UI *calls* a Groq-hosted model with your key — not that
tinygpt runs in the cloud. Use de-identified text only; the conversation leaves your network.

## Run

```bash
cd llm
pip install -r webui/requirements.txt        # flask + httpx (torch/numpy already needed by the model)
cp .env.example .env                          # then paste your GROQ_API_KEY into .env
python run_webui.py                           # -> http://127.0.0.1:8080
```

- **Groq** works as soon as a valid `GROQ_API_KEY` is in `.env`. The model dropdown is filled
  from Groq's live `/models` list, so it never depends on a hard-coded id.
- **tinygpt** needs a trained checkpoint. Train one first (`python train.py`), or point
  `TINYGPT_CKPT` at an existing `.pt`. Until then the tinygpt backend reports that it has no
  checkpoint; Groq still works.

All settings (key, model, device, host/port, generation defaults) are environment variables —
see `../.env.example`. `.env` is git-ignored, so your key is never committed.

## Genetics lookups (Groq backend)

The Groq model gets three lookup tools and instructions to use them for any specific fact,
so classifications and frequencies come from data, not from the model's memory. Each lookup
shows above the answer with links to the source record.

| Tool | Answers from | Where |
|------|--------------|-------|
| `lookup_variant` | gene, consequence, HGVS c./p. (MANE), exon, AlphaMissense, PubMed IDs | **local** VEP 116 in WSL |
| | position check: left-aligned against GRCh38, wrong REF base caught | **local** bcftools + FASTA |
| | ClinVar classification, stars, conditions; P/LP variants at the same amino acid / within 15 bp | **local** copy |
| | population frequencies | gnomAD v4.1 API (online: counts, homozygotes, FAF95), else VEP cache (**local**: AFs) |
| | inside the lab's exome capture kit? | **local** BED |
| `lookup_gene` | gnomAD v4.1 constraint (pLI, LOEUF, missense z) | **local** |
| | ClinGen **dosage sensitivity** (HI/TS scores) - not gene-disease validity | **local** |
| | linked diseases + inheritance (HPO/OMIM/Orphanet), ClinVar counts | **local** |
| `lookup_phenotype` | HPO term, definition, synonyms, annotated genes | **local** |

Online calls send only the gene symbol or variant name: NCBI (to identify a variant from
its name, e.g. `BRCA1:c.68_69del` -> ClinVar ID), gnomAD (frequencies), and Ensembl (only
for an rsID that is on no local ClinVar record). Set `GENETICS_ONLINE=false` to stay local:
everything still works except finding a variant by HGVS *name* (give a position, rsID or
ClinVar ID instead). VEP runs through `wsl.exe` using the AI-Analyser's tools
(`webui/knowledge/vep_annotate.py`, ~1 s per lookup); `GENETICS_VEP=false` turns it off.

**Build the local database** (once, and again after the AI-Analyser resources are updated).
The resources live inside WSL, so the builder runs there and writes to `llm/data/`:

```bash
wsl -d Ubuntu -u root -- python3 \
  /mnt/c/Users/Anderson/Desktop/Projects/LLM_development/llm/webui/knowledge/build_db.py \
  --resources /root/wes/resources \
  --out /mnt/c/Users/Anderson/Desktop/Projects/LLM_development/llm/data/genetics.sqlite
```

About 1 minute; ~1.3 GB (git-ignored). Restart the UI afterwards.

**ClinGen gene-disease validity** (Definitive/Strong/Moderate/... per gene-disease pair) is a
small public file read directly. Download or refresh it from `llm/`:

```bash
.venv\Scripts\python.exe -m webui.knowledge.fetch_public   # -> data/clingen_gene_validity.csv
```

Limits worth knowing: lookups fix *facts*, not *reasoning* - general answers (e.g. ACMG
rule details) still come from the model's memory and can be wrong. Gene aliases are not
resolved (use `GBA1`, not `GBA`). Groq's free tier allows 8,000 tokens/minute on
gpt-oss-120b; a lookup answer uses ~3-4k, so expect a short wait after 2 quick questions.

## How it works

- `webui/config.py` — settings + a tiny `.env` loader.
- `webui/backends.py` — `GroqBackend` (streams from Groq's API) and `TinyGPTBackend` (loads
  the checkpoint once, generates one token at a time under a lock so it can stream and so
  concurrent tabs take turns). Both expose the same `stream(messages, params)`.
- `webui/app.py` — Flask: `/` (page), `/api/chat` (SSE stream), `/api/health`, `/api/models`.
- `webui/static/` + `webui/templates/` — the single-page chat UI; replies stream in live.
