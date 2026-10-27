"""The AI backend adapter: both backends must honour the same contract, and any
failure must raise AIError so the pipeline falls back to the rule-based draft."""
from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from wes import ai
from wes.config import settings


class _Out(ai._Strict):
    flag: str
    notes: list[str]


def _reply(payload: dict, status: int = 200):
    """Patch httpx.post with a canned Ollama /api/chat response."""
    def fake_post(url, **kw):
        fake_post.seen = {"url": url, **kw}
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))
    return fake_post


@pytest.fixture
def local(monkeypatch):
    """Settings is a frozen dataclass, so swap in a copy rather than mutating it."""
    monkeypatch.setattr(ai, "settings", replace(settings, ai_backend="local", ai_enabled=True,
                                                ollama_model="qwen3.5:4b"))


def test_local_backend_parses_and_reports_usage(local, monkeypatch):
    post = _reply({"model": "qwen3.5:4b", "done_reason": "stop",
                   "message": {"content": json.dumps({"flag": "NEGATIVE", "notes": ["ok"]})},
                   "prompt_eval_count": 1200, "eval_count": 80, "total_duration": 42_000_000_000})
    monkeypatch.setattr(httpx, "post", post)
    out, usage = ai._call("sys", "user", _Out, effort="high")
    assert out.flag == "NEGATIVE" and out.notes == ["ok"]
    assert usage["backend"] == "local" and usage["input_tokens"] == 1200 and usage["duration_s"] == 42.0

    sent = post.seen["json"]
    assert sent["options"]["temperature"] == 0          # clinical output must be reproducible
    assert sent["stream"] is False
    assert sent["format"]["properties"].keys() == {"flag", "notes"}  # schema constrains generation
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


@pytest.mark.parametrize("payload,status,fragment", [
    ({"message": {"content": "not json"}}, 200, "valid JSON"),
    ({"message": {"content": json.dumps({"flag": "X"})}}, 200, "did not match"),
    ({"message": {"content": ""}}, 200, "empty"),
    ({"message": {"content": "{}"}, "done_reason": "length", "eval_count": 8000}, 200, "did not stop"),
    ({"error": "model not found"}, 404, "ollama pull"),
    ({"error": "boom"}, 500, "500"),
])
def test_local_backend_failures_raise_aierror(local, monkeypatch, payload, status, fragment):
    monkeypatch.setattr(httpx, "post", _reply(payload, status))
    with pytest.raises(ai.AIError, match=fragment):
        ai._call("sys", "user", _Out, effort="high")


def test_unreachable_server_raises_aierror(local, monkeypatch):
    def refuse(url, **kw):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ai.AIError, match="not reachable"):
        ai._call("sys", "user", _Out, effort="high")


def test_available_checks_the_model_is_installed(local, monkeypatch):
    def tags(names):
        return lambda url, **kw: httpx.Response(
            200, json={"models": [{"name": n} for n in names]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", tags(["qwen3.5:4b", "gemma4:e4b"]))
    assert ai.available() is True
    monkeypatch.setattr(httpx, "get", tags(["gemma4:e4b"]))
    assert ai.available() is False

    def down(url, **kw):
        raise httpx.ConnectError("refused", request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", down)
    assert ai.available() is False


def test_default_backend_is_local_and_claude_needs_opting_in():
    # a fresh install must never send case data off-site
    assert settings.ai_backend == "local"
    assert ai.backend() == "local"
    assert "local model" in ai.describe_backend()


def test_claude_backend_not_imported_unless_selected(local, monkeypatch):
    """The anthropic package is optional: using the local backend must not need it."""
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", None)   # make importing it fail
    monkeypatch.setattr(httpx, "post", _reply(
        {"message": {"content": json.dumps({"flag": "NEGATIVE", "notes": []})}, "done_reason": "stop"}))
    assert ai._call("sys", "user", _Out, effort="high")[0].flag == "NEGATIVE"
