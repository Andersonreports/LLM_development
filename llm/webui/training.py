"""Run training as a background job so the browser can start it and watch it live.

One job at a time (a second would fight the first for the GPU). We shell out to the
project's own train.py / prepare_data.py - the UI never reimplements training, it just
drives the scripts and tails their output. The live loss curve is read from the run's
log.csv (written by train.py); the sample text and status come from the captured stdout.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
UPLOAD_DIR = ROOT / "data" / "_uploads"

_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(raw: str, fallback: str) -> str:
    name = _NAME_RE.sub("_", (raw or "").strip())[:60].strip("._")
    return name or fallback


# --------------------------------------------------------------------------- datasets
def list_datasets() -> list[dict]:
    """Prepared datasets: a data/<name>/ folder with train.bin + tokenizer.json."""
    out: list[dict] = []
    if DATA_DIR.exists():
        for d in sorted(DATA_DIR.iterdir()):
            if d.name.startswith("_") or not d.is_dir():
                continue
            if not (d / "train.bin").exists() or not (d / "tokenizer.json").exists():
                continue
            info = {"name": d.name, "path": str(d)}
            meta = d / "meta.json"
            if meta.exists():
                try:
                    import json
                    m = json.loads(meta.read_text())
                    info["tokens_train"] = m.get("tokens_train")
                    info["vocab_size"] = m.get("vocab_size")
                    info["tokenizer"] = m.get("tokenizer")
                except Exception:  # noqa: BLE001
                    pass
            out.append(info)
    return out


def prepare_dataset(*, corpus: str | None = None, text_path: Path | None = None,
                    name: str | None = None, tokenizer: str = "char") -> dict:
    """Run prepare_data.py for either the built-in corpus or an uploaded text file.
    Blocking (preparation is quick for ~1 MB); returns the dataset info or raises."""
    cmd = [sys.executable, "prepare_data.py", "--tokenizer", tokenizer]
    if text_path is not None:
        ds_name = _safe_name(name or text_path.stem, "mine")
        out = DATA_DIR / ds_name
        cmd += ["--text", str(text_path), "--out", str(out)]
    else:
        cmd += ["--corpus", corpus or "shakespeare"]
        ds_name = (corpus or "shakespeare") + ("_bpe" if tokenizer == "bpe" else "")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "prepare_data failed").strip()[-800:])
    return {"name": ds_name, "stdout": p.stdout.strip()[-800:]}


# --------------------------------------------------------------------------- training job
@dataclass
class Job:
    name: str
    dataset: str
    preset: str
    max_iters: int
    device: str
    started_at: float = field(default_factory=time.time)
    proc: subprocess.Popen | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=500))
    returncode: int | None = None
    sample: str = ""


class Trainer:
    def __init__(self) -> None:
        self._job: Job | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._job is not None and self._job.proc is not None and self._job.proc.poll() is None

    def start(self, *, dataset: str, name: str, preset: str, max_iters: int, device: str) -> dict:
        with self._lock:
            if self.is_running():
                raise RuntimeError("a training run is already in progress")
            ds = DATA_DIR / dataset
            if not (ds / "train.bin").exists():
                raise RuntimeError(f"dataset {dataset!r} is not prepared (no train.bin)")
            run_name = _safe_name(name, dataset + "_run")
            # evaluate ~20 times over the run so the live loss curve has points to draw
            eval_interval = max(5, int(max_iters) // 20)
            cmd = [sys.executable, "train.py",
                   "--data", str(ds), "--name", run_name,
                   "--preset", preset, "--max-iters", str(int(max_iters)),
                   "--eval-interval", str(eval_interval),
                   "--out-dir", str(CKPT_DIR)]
            if device and device != "auto":
                cmd += ["--device", device]
            # line-buffered, stderr folded into stdout so errors show in the live log
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            job = Job(name=run_name, dataset=dataset, preset=preset,
                      max_iters=int(max_iters), device=device or "auto", proc=proc)
            self._job = job
            threading.Thread(target=self._pump, args=(job,), name="train-pump", daemon=True).start()
            return {"name": run_name, "started": True}

    def _pump(self, job: Job) -> None:
        assert job.proc and job.proc.stdout
        for line in job.proc.stdout:
            line = line.rstrip("\n")
            job.lines.append(line)
            if "sample:" in line:
                job.sample = line.split("sample:", 1)[1].strip()
        job.returncode = job.proc.wait()

    def stop(self) -> dict:
        with self._lock:
            if self._job and self._job.proc and self._job.proc.poll() is None:
                self._job.proc.terminate()
                try:
                    self._job.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._job.proc.kill()
                return {"stopped": True}
            return {"stopped": False}

    def _curve(self, run_name: str, max_points: int = 120) -> list[dict]:
        log = CKPT_DIR / run_name / "log.csv"
        if not log.exists():
            return []
        rows: list[dict] = []
        try:
            with open(log, newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append({"iter": int(r["iter"]),
                                     "val": float(r["val_loss"]),
                                     "train": float(r["train_loss"])})
                    except (KeyError, ValueError):
                        continue
        except OSError:
            return []
        if len(rows) > max_points:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)]
        return rows

    def status(self) -> dict:
        job = self._job
        if job is None:
            return {"state": "idle"}
        running = self.is_running()
        curve = self._curve(job.name)
        last = curve[-1] if curve else None
        return {
            "state": "running" if running else "done",
            "name": job.name,
            "dataset": job.dataset,
            "preset": job.preset,
            "device": job.device,
            "max_iters": job.max_iters,
            "elapsed": round(time.time() - job.started_at, 1),
            "returncode": job.returncode,
            "iter": last["iter"] if last else None,
            "val": last["val"] if last else None,
            "train": last["train"] if last else None,
            "curve": curve,
            "sample": job.sample,
            "log": list(job.lines)[-40:],
        }


trainer = Trainer()
