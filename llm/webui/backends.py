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
import re
import sys
import threading
import time
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
        ok, why = self.tools_status()
        if ok:
            try:
                from .knowledge import lookup
                tools_txt = (f"genetics lookups ON (ClinVar {lookup.meta().get('clinvar_date')}, "
                             f"gnomAD, ClinGen, HPO)")
            except Exception:  # noqa: BLE001
                tools_txt = "genetics lookups ON"
        else:
            tools_txt = f"genetics lookups {why}"
        return (f"Groq Cloud ({settings.groq_model}, {key}) · {tools_txt} - "
                f"conversation is sent to Groq's servers")

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
            # Return every model; the UI labels each one (chat / voice / ...) and lists
            # chat models first, so non-chat entries are visible but clearly marked.
            return sorted(m["id"] for m in r.json().get("data", []))
        except Exception:  # noqa: BLE001 - network/key problems -> empty, UI falls back
            return []

    # Most lookup rounds a single answer may take before it must reply with what it has.
    MAX_TOOL_ROUNDS = 5
    # gpt-oss spends part of max_tokens on hidden reasoning, so tool mode never goes below
    # these. The first round usually just decides on lookups; the rounds after it write the
    # answer from the results (an ACMG table needs room). Groq's free tier counts
    # max_tokens against 8,000 tokens/minute, so the first round stays small.
    FIRST_ROUND_TOKENS = 1500
    ANSWER_TOKENS = 3000
    # Groq's 429 says how long to wait; wait it out automatically only if it is short.
    MAX_RATE_WAIT = 20

    def tools_status(self) -> tuple[bool, str]:
        if not settings.genetics_tools:
            return False, "off (GENETICS_TOOLS=false)"
        from .knowledge import tools
        return tools.available()

    def stream(self, messages: list[dict], params: GenParams,
               model: str | None = None) -> Iterator[str | dict]:
        """Yields text chunks (str) and, in tool mode, lookup events ({"tool": {...}})."""
        ok, reason = self.available()
        if not ok:
            raise BackendError(reason)
        use_tools, _ = self.tools_status()
        msgs = list(messages)
        max_tokens = params.max_tokens
        if use_tools:
            from .knowledge import tools
            if not any(m.get("role") == "system" for m in msgs):
                msgs.insert(0, {"role": "system", "content": tools.SYSTEM_PROMPT})

        for round_no in range(self.MAX_TOOL_ROUNDS + 1):
            if use_tools:
                floor = self.FIRST_ROUND_TOKENS if round_no == 0 else self.ANSWER_TOKENS
                max_tokens = max(params.max_tokens, floor)
            body = {
                "model": model or settings.groq_model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": params.temperature,
                "stream": True,
            }
            if params.top_p is not None:
                body["top_p"] = params.top_p
            # On the last round no tools are offered, so the model has to answer.
            if use_tools and round_no < self.MAX_TOOL_ROUNDS:
                body["tools"] = tools.TOOLS
                body["tool_choice"] = "auto"

            text: list[str] = []
            calls: dict[int, dict] = {}
            finish = None
            for attempt in range(2):
                try:
                    for kind, value in self._stream_once(body):
                        if kind == "text":
                            text.append(value)
                            yield value
                        elif kind == "finish":
                            finish = value
                        else:  # a fragment of a tool call; arguments arrive in pieces
                            idx = value.get("index", 0)
                            c = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            c["id"] = value.get("id") or c["id"]
                            fn = value.get("function") or {}
                            c["name"] = fn.get("name") or c["name"]
                            c["arguments"] += fn.get("arguments") or ""
                    break
                except _ToolCallRejected as e:
                    # Groq rejected a malformed lookup call before any answer text; the
                    # model usually gets it right on a second try.
                    if attempt == 0 and not text:
                        calls.clear()
                        continue
                    raise BackendError(f"the model made an invalid lookup call twice ({e})") \
                        from None
                except _RateLimited as e:
                    # Rejected before any output, so a retry is safe. Wait only if short.
                    if attempt == 0 and e.wait <= self.MAX_RATE_WAIT:
                        yield {"notice": f"Groq free-tier limit reached - waiting "
                                         f"{e.wait:.0f} s, then continuing…"}
                        time.sleep(e.wait + 0.5)
                        continue
                    raise BackendError(
                        f"Groq's per-minute limit for this model is used up ({e.detail}). "
                        f"Wait about {max(e.wait, 1):.0f} s and send again.") from None
            if not calls:
                if finish == "length":  # never let a cut-off answer look complete
                    yield {"notice": "Answer cut off at the length limit. Ask \"continue\", "
                                     "or raise Max tokens in ⚙ settings."}
                    yield "\n\n⚠ [answer cut off at the length limit]"
                return

            ordered = [calls[i] for i in sorted(calls)]
            msgs.append({"role": "assistant", "content": "".join(text) or None,
                         "tool_calls": [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"],
                                                      "arguments": c["arguments"] or "{}"}}
                                        for c in ordered]})
            for c in ordered:
                yield {"tool": {"name": c["name"], "status": "running",
                                "query": _first_arg(c["arguments"])}}
                result, event = tools.run(c["name"], c["arguments"])
                yield {"tool": {**event, "status": "done"}}
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result, ensure_ascii=False,
                                                   separators=(",", ":"))[:8000]})

    def _stream_once(self, body: dict) -> Iterator[tuple[str, object]]:
        """One streamed Groq request -> ("text", str) and ("call", tool_call_delta) items."""
        try:
            with httpx.stream("POST", f"{settings.groq_base_url}/chat/completions",
                              headers=self._headers(), json=body,
                              timeout=settings.request_timeout) as r:
                if r.status_code == 429:
                    raise _RateLimited.from_response(r)
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
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:  # Groq can fail mid-stream; never swallow it
                        err = chunk["error"] if isinstance(chunk["error"], dict) else {}
                        if err.get("code") == "tool_use_failed":
                            raise _ToolCallRejected(err.get("message", "tool call rejected"))
                        raise BackendError(f"Groq error: {err.get('message') or chunk['error']}")
                    try:
                        delta = chunk["choices"][0]["delta"]
                    except (KeyError, IndexError):
                        continue
                    if delta.get("content"):
                        yield "text", delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        yield "call", tc
                    reason = chunk["choices"][0].get("finish_reason")
                    if reason:
                        yield "finish", reason
        except httpx.HTTPError as e:
            raise BackendError(f"could not reach Groq: {e}") from e


class _ToolCallRejected(Exception):
    """Groq refused the model's tool call (bad arguments) - code 'tool_use_failed'."""


class _RateLimited(Exception):
    """Groq answered 429 (tokens/requests per minute used up)."""

    def __init__(self, wait: float, detail: str):
        super().__init__(detail)
        self.wait, self.detail = wait, detail

    @classmethod
    def from_response(cls, r) -> "_RateLimited":
        body = r.read().decode("utf-8", "replace")
        try:
            msg = json.loads(body)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = body[:300]
        wait = None
        try:
            wait = float(r.headers.get("retry-after", ""))
        except ValueError:
            pass
        if wait is None:  # "... Please try again in 7.53s." / "in 1m2.5s"
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 30.0
        short = re.search(r"\((TPM|RPM|TPD|RPD)\): Limit [\d,]+", msg)
        return cls(wait, short.group(0) if short else "rate limit")


def _first_arg(raw: str) -> str:
    try:
        args = json.loads(raw or "{}")
        return str(next(iter(args.values()), "")) if isinstance(args, dict) else ""
    except json.JSONDecodeError:
        return ""


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
