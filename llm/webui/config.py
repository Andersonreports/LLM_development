"""Settings, all driven by environment variables (optionally from a .env file next to
run_webui.py). Nothing secret is hard-coded; the Groq key only ever comes from the
environment. Import this module early - it loads .env on import."""
from __future__ import annotations

import os
from pathlib import Path

# llm/  (the project root: holds tinygpt/, generate.py, checkpoints/, ...)
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=value per line, # comments, optional quotes). Values
    already present in the real environment win, so you can always override a .env line."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # Which backend answers by default. The browser can switch per-message, so this is
    # only the initial choice. "groq" or "tinygpt".
    backend: str = os.environ.get("LLM_BACKEND", "groq").strip().lower()

    # ---- Groq (OpenAI-compatible) ----
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "").strip()
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    # Default model id. Groq's catalogue changes often, so the UI also fetches the live
    # list from /models; if this id is gone, pick another from the dropdown.
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    # ---- genetics knowledge (Groq only) ----
    # Local SQLite built by webui/knowledge/build_db.py from the AI-Analyser resources.
    genetics_db: Path = Path(os.environ.get(
        "GENETICS_DB", str(ROOT / "data" / "genetics.sqlite")))
    # Give the Groq model the lookup tools + the genetics system prompt.
    genetics_tools: bool = _bool("GENETICS_TOOLS", True)
    # Allow the two online lookups (NCBI name search, gnomAD variant frequencies).
    genetics_online: bool = _bool("GENETICS_ONLINE", True)
    # Local VEP (+ AlphaMissense, gnomAD frequencies from the cache, PubMed IDs) runs
    # inside WSL with the AI-Analyser's tools and resources.
    genetics_vep: bool = _bool("GENETICS_VEP", True)
    wsl_distro: str = os.environ.get("GENETICS_WSL_DISTRO", "Ubuntu")
    wsl_resources: str = os.environ.get("GENETICS_WSL_RESOURCES", "/root/wes/resources")
    wsl_tools_bin: str = os.environ.get("GENETICS_WSL_TOOLS_BIN",
                                        "/root/wes/tools/mamba/envs/wes/bin")

    # ---- tinygpt (our own model) ----
    tinygpt_ckpt: Path = Path(os.environ.get(
        "TINYGPT_CKPT", str(ROOT / "checkpoints" / "shakespeare_char" / "best.pt")))
    # Empty -> auto (cuda if available, else cpu); resolved in backends.py so torch is not
    # imported just to read settings.
    tinygpt_device: str = os.environ.get("TINYGPT_DEVICE", "").strip()
    # Load the checkpoint at startup in a background thread so the first message is fast.
    tinygpt_warm_start: bool = _bool("TINYGPT_WARM_START", True)

    # ---- generation defaults (the browser can override) ----
    max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "512"))
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.8"))
    top_k: int = int(os.environ.get("LLM_TOP_K", "40"))

    # ---- server ----
    host: str = os.environ.get("LLM_HOST", "127.0.0.1")
    port: int = int(os.environ.get("LLM_PORT", "8080"))
    # HTTP timeout (seconds) for the Groq API.
    request_timeout: float = float(os.environ.get("LLM_REQUEST_TIMEOUT", "120"))


settings = Settings()
