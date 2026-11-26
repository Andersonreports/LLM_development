"""The two backends behind the chat UI, with one shared shape:

    available() -> (ok, reason)   can this backend answer right now?
    describe()  -> str            one line for the UI / status endpoint
    stream(messages, params)      yield output text chunk by chunk

`messages` is the OpenAI-style list [{"role": "user"|"assistant"|"system", "content": str}].
`params` is a GenParams. Backends are created once and reused (the tinygpt model is loaded
a single time and kept in memory); generation is serialised with a lock so concurrent
browser tabs take turns on the one model instead of corrupting each other's state.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import httpx

from .config import ROOT, settings

# Make `import tinygpt` work no matter where the server was launched from.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BackendError(RuntimeError):
    """Raised when a backend cannot produce output; the message is shown to the user."""


@dataclass
class GenParams:
    max_tokens: int = settings.max_tokens
    temperature: float = settings.temperature
    top_k: int | None = settings.top_k
    top_p: float | None = None

    @classmethod
    def from_request(cls, data: dict) -> "GenParams":
        def num(key, default, cast):
            v = data.get(key, default)
            try:
                return cast(v) if v is not None and v != "" else default
            except (TypeError, ValueError):
                return default
        return cls(
            max_tokens=max(1, min(4096, num("max_tokens", settings.max_tokens, int))),
            temperature=max(0.0, min(2.0, num("temperature", settings.temperature, float))),
            top_k=num("top_k", settings.top_k, int) or None,
            top_p=num("top_p", None, float),
        )


# --------------------------------------------------------------------------- Groq
class GroqBackend:
    name = "groq"
    kind = "chat"  # understands the full conversation and system/role turns

    def available(self) -> tuple[bool, str]:
        if not settings.groq_api_key:
            return False, "GROQ_API_KEY is not set (put it in llm/.env)."
        return True, "ready"

    def describe(self) -> str:
        key = "key set" if settings.groq_api_key else "no key"
        return f"Groq Cloud ({settings.groq_model}, {key}) - conversation is sent to Groq's servers"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        """Live catalogue from Groq, so the dropdown never relies on a hard-coded id."""
        if not settings.groq_api_key:
            return []
        try:
            r = httpx.get(f"{settings.groq_base_url}/models", headers=self._headers(),
                          timeout=15)
            r.raise_for_status()
            ids = sorted(m["id"] for m in r.json().get("data", []))
            # Keep text chat models near the top; drop audio/whisper/guard-only entries.
            chatty = [m for m in ids if not any(x in m.lower()
                      for x in ("whisper", "tts", "guard", "embed"))]
            return chatty or ids
        except Exception:  # noqa: BLE001 - network/key problems -> empty, UI falls back
            return []

    def stream(self, messages: list[dict], params: GenParams,
               model: str | None = None) -> Iterator[str]:
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        body = {
            "model": model or settings.groq_model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": True,
        }
        if params.top_p is not None:
            body["top_p"] = params.top_p
        try:
            with httpx.stream("POST", f"{settings.groq_base_url}/chat/completions",
                              headers=self._headers(), json=body,
                              timeout=settings.request_timeout) as r:
                if r.status_code != 200:
                    detail = r.read().decode("utf-8", "replace")[:500]
                    raise BackendError(f"Groq returned HTTP {r.status_code}: {detail}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise BackendError(f"could not reach Groq: {e}") from e


# --------------------------------------------------------------------------- tinygpt
_GEN_LOCK = threading.Lock()  # one generation at a time on the single model instance


class TinyGPTBackend:
    name = "tinygpt"
    kind = "completion"  # continues the text you give it; no notion of chat roles

    def __init__(self) -> None:
        self._model = None
        self._tok = None
        self._device = None
        self._load_lock = threading.Lock()
        self._load_error: str | None = None

    # -- loading -----------------------------------------------------------
    def _resolve_device(self) -> str:
        if settings.tinygpt_device:
            return settings.tinygpt_device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_loaded(self) -> None:
        """Load the checkpoint once. Safe to call from several threads; the first wins."""
        if self._model is not None or self._load_error is not None:
            return
        with self._load_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                if not settings.tinygpt_ckpt.exists():
                    raise BackendError(
                        f"no checkpoint at {settings.tinygpt_ckpt} - train one with "
                        f"`python train.py`, or set TINYGPT_CKPT.")
                import torch  # noqa: F401  (imported here to keep startup light)
                from generate import load as _load  # reuse the tested loader
                device = self._resolve_device()
                model, tok = _load(settings.tinygpt_ckpt, device)
                self._model, self._tok, self._device = model, tok, device
            except Exception as e:  # noqa: BLE001
                self._load_error = str(e)

    def warm_start(self) -> None:
        """Kick off loading in the background so the first message isn't slow."""
        threading.Thread(target=self.ensure_loaded, name="tinygpt-load", daemon=True).start()

    def set_checkpoint(self, path: Path) -> None:
        """Switch to a different trained checkpoint (from the history panel). Drops the
        loaded model so the next message loads the newly chosen one."""
        with self._load_lock:
            settings.tinygpt_ckpt = path
            self._model = self._tok = self._device = None
            self._load_error = None

    def available(self) -> tuple[bool, str]:
        if self._load_error:
            return False, self._load_error
        if self._model is not None:
            return True, "ready"
        if not settings.tinygpt_ckpt.exists():
            return False, (f"no checkpoint at {settings.tinygpt_ckpt} - train one with "
                           f"`python train.py`, or set TINYGPT_CKPT.")
        return True, "checkpoint found (loads on first use)"

    def describe(self) -> str:
        where = "loaded" if self._model is not None else "not yet loaded"
        dev = self._device or (settings.tinygpt_device or "auto")
        return (f"tinygpt {settings.tinygpt_ckpt.name} on {dev} ({where}) - runs locally, "
                f"completion only (continues your text)")

    # -- generation --------------------------------------------------------
    @staticmethod
    def _prompt_from(messages: list[dict]) -> str:
        """A completion model has no chat format, so we seed it with the latest thing the
        user typed (falling back to a newline)."""
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                return m["content"]
        return "\n"

    def _encode(self, text: str) -> list[int]:
        text = text or "\n"
        try:
            return self._tok.encode(text)
        except ValueError:
            # CharTokenizer rejects characters it never saw in training; keep the ones it
            # knows so a stray emoji doesn't abort the whole request.
            stoi = getattr(self._tok, "stoi", None)
            if stoi is None:
                raise
            kept = "".join(c for c in text if c in stoi) or "\n"
            return self._tok.encode(kept)

    def stream(self, messages: list[dict], params: GenParams,
               model: str | None = None) -> Iterator[str]:
        self.ensure_loaded()
        ok, reason = self.available()
        if not ok or self._model is None:
            raise BackendError(reason)

        import torch
        prompt_ids = self._encode(self._prompt_from(messages))
        with _GEN_LOCK:
            idx = torch.tensor([prompt_ids], device=self._device)
            generated: list[int] = []
            shown = ""
            for _ in range(params.max_tokens):
                # One token at a time reuses the model's own tested sampling and lets us
                # stream; decoding the whole generated run each step keeps multi-byte
                # (BPE) output correct.
                idx = self._model.generate(
                    idx, 1, temperature=params.temperature,
                    top_k=params.top_k, top_p=params.top_p)
                generated.append(int(idx[0, -1].item()))
                text = self._tok.decode(generated)
                delta, shown = text[len(shown):], text
                if delta:
                    yield delta


# --------------------------------------------------------------------------- registry
_INSTANCES: dict[str, object] = {"groq": GroqBackend(), "tinygpt": TinyGPTBackend()}


def get_backend(name: str | None):
    key = (name or settings.backend or "groq").strip().lower()
    if key not in _INSTANCES:
        raise BackendError(f"unknown backend {key!r}; choose 'groq' or 'tinygpt'.")
    return _INSTANCES[key]


def all_status() -> list[dict]:
    out = []
    for key in ("groq", "tinygpt"):
        b = _INSTANCES[key]
        ok, reason = b.available()
        out.append({"name": key, "kind": b.kind, "available": ok,
                    "reason": reason, "describe": b.describe()})
    return out
