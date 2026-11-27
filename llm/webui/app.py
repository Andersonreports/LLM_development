"""Flask app: serves the chat page and a streaming chat endpoint.

Run it with `python run_webui.py` from the llm/ directory.
"""
from __future__ import annotations

import json
from typing import Iterator

from flask import Flask, Response, jsonify, render_template, request

from pathlib import Path

from .backends import GenParams, BackendError, all_status, get_backend
from .config import settings
from .runs import list_runs, resolve_run_path
from .training import (UPLOAD_DIR, list_datasets, prepare_dataset, trainer, _safe_name)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload cap


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return render_template("index.html", default_backend=settings.backend)


@app.get("/api/health")
def health():
    return jsonify({"backends": all_status(), "default": settings.backend})


@app.get("/api/models")
def models():
    """Live Groq model list for the dropdown (empty if no key / offline)."""
    name = request.args.get("backend", "groq")
    if name == "groq":
        return jsonify({"models": get_backend("groq").list_models(),
                        "default": settings.groq_model})
    if name == "ollama":
        return jsonify({"models": get_backend("ollama").list_models(),
                        "default": settings.ollama_model})
    # tinygpt has exactly one "model": the loaded checkpoint.
    return jsonify({"models": [settings.tinygpt_ckpt.name], "default": settings.tinygpt_ckpt.name})


@app.get("/api/runs")
def runs():
    """History of trained models (checkpoints under llm/checkpoints/)."""
    return jsonify(list_runs())


@app.post("/api/use_ckpt")
def use_ckpt():
    """Point the tinygpt backend at one of the trained checkpoints from the history."""
    data = request.get_json(silent=True) or {}
    try:
        path = resolve_run_path(data.get("path", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    backend = get_backend("tinygpt")
    backend.set_checkpoint(path)
    ok, reason = backend.available()
    return jsonify({"active": str(path), "available": ok, "reason": reason,
                    "describe": backend.describe()})


@app.get("/api/datasets")
def datasets():
    return jsonify({"datasets": list_datasets()})


@app.post("/api/prepare")
def prepare():
    """Prepare a dataset: either the built-in corpus, or an uploaded .txt file."""
    try:
        if "file" in request.files and request.files["file"].filename:
            f = request.files["file"]
            tokenizer = request.form.get("tokenizer", "char")
            name = request.form.get("name") or Path(f.filename).stem
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / (_safe_name(name, "corpus") + ".txt")
            f.save(dest)
            info = prepare_dataset(text_path=dest, name=name, tokenizer=tokenizer)
        else:
            data = request.get_json(silent=True) or {}
            info = prepare_dataset(corpus=data.get("corpus", "shakespeare"),
                                   tokenizer=data.get("tokenizer", "char"))
        return jsonify({"ok": True, **info})
    except Exception as e:  # noqa: BLE001 - surface the failure to the browser
        return jsonify({"error": str(e)}), 400


@app.post("/api/train/start")
def train_start():
    data = request.get_json(silent=True) or {}
    try:
        res = trainer.start(
            dataset=data.get("dataset", ""),
            name=data.get("name", ""),
            preset=data.get("preset", "tiny"),
            max_iters=int(data.get("max_iters", 500)),
            device=data.get("device", "auto"),
        )
        return jsonify(res)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.get("/api/train/status")
def train_status():
    return jsonify(trainer.status())


@app.post("/api/train/stop")
def train_stop():
    return jsonify(trainer.stop())


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    backend_name = data.get("backend")
    model = data.get("model")
    params = GenParams.from_request(data)

    try:
        backend = get_backend(backend_name)
    except BackendError as e:
        return jsonify({"error": str(e)}), 400

    def produce() -> Iterator[str]:
        yield _sse({"meta": {"backend": backend.name, "kind": backend.kind}})
        try:
            for chunk in backend.stream(messages, params, model=model):
                # str = answer text; dict = a lookup event ({"tool": {...}}) for the page
                yield _sse(chunk if isinstance(chunk, dict) else {"delta": chunk})
        except BackendError as e:
            yield _sse({"error": str(e)})
        except Exception as e:  # noqa: BLE001 - never leak a traceback to the browser
            yield _sse({"error": f"{type(e).__name__}: {e}"})
        yield _sse({"done": True})

    return Response(produce(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main() -> None:
    # Warm up tinygpt in the background so the first local message isn't slow.
    if settings.tinygpt_warm_start:
        try:
            get_backend("tinygpt").warm_start()
        except Exception:  # noqa: BLE001 - never block startup on the optional model
            pass
    print(f"ADVAT LLM chat UI -> http://{settings.host}:{settings.port}  "
          f"(default backend: {settings.backend})")
    app.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
