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

## How it works

- `webui/config.py` — settings + a tiny `.env` loader.
- `webui/backends.py` — `GroqBackend` (streams from Groq's API) and `TinyGPTBackend` (loads
  the checkpoint once, generates one token at a time under a lock so it can stream and so
  concurrent tabs take turns). Both expose the same `stream(messages, params)`.
- `webui/app.py` — Flask: `/` (page), `/api/chat` (SSE stream), `/api/health`, `/api/models`.
- `webui/static/` + `webui/templates/` — the single-page chat UI; replies stream in live.
