"""Discover past training runs so the UI can show a history of the models you've trained.

A "run" is one checkpoint directory under llm/checkpoints/<name>/, as written by train.py:
  best.pt / last.pt  weights + {config, train_args, iter, best_val, tokenizer}
  log.csv            iter,train_loss,val_loss,lr,elapsed_s,gpu_mem_gb  (the loss curve)
  summary.json       {iter, train, val, best_val, minutes}            (final numbers)

Reading the .pt is the only expensive part, so each file's extracted facts are cached by
(path, mtime) - re-listing is then cheap, and params are summed straight from the saved
tensors (exact, no model rebuild)."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .config import ROOT, settings

CKPT_ROOT = ROOT / "checkpoints"

_META_CACHE: dict[tuple[str, float], dict] = {}


def _ckpt_meta(pt: Path) -> dict:
    """Small facts from a .pt (config, arch, size), cached by modification time."""
    key = (str(pt), pt.stat().st_mtime)
    if key in _META_CACHE:
        return _META_CACHE[key]
    meta: dict = {}
    try:
        import torch
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or {}
        args = ck.get("train_args", {}) or {}
        params = int(sum(t.numel() for t in ck.get("model", {}).values()))
        meta = {
            "params": params,
            "config": {k: cfg.get(k) for k in
                       ("n_layer", "n_head", "n_embd", "block_size", "vocab_size")},
            "iter": ck.get("iter"),
            "best_val": ck.get("best_val"),
            "device": args.get("device"),
            "dataset": args.get("name") or args.get("dataset"),
        }
    except Exception as e:  # noqa: BLE001 - a corrupt/old checkpoint shouldn't break the list
        meta = {"error": str(e)}
    _META_CACHE[key] = meta
    return meta


def _curve(log: Path, max_points: int = 60) -> list[dict]:
    """val/train loss per eval step from log.csv, down-sampled for a sparkline."""
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


def _active_path() -> str:
    try:
        return str(settings.tinygpt_ckpt.resolve())
    except OSError:
        return str(settings.tinygpt_ckpt)


def list_runs() -> dict:
    """All training runs, newest first, with the metrics the UI needs."""
    runs: list[dict] = []
    active = _active_path()
    if CKPT_ROOT.exists():
        for d in sorted(CKPT_ROOT.iterdir()):
            if not d.is_dir():
                continue
            best, last = d / "best.pt", d / "last.pt"
            primary = best if best.exists() else last
            if not primary.exists():
                continue  # a directory with no weights isn't a run

            summary = {}
            sj = d / "summary.json"
            if sj.exists():
                try:
                    summary = json.loads(sj.read_text())
                except (OSError, json.JSONDecodeError):
                    summary = {}

            meta = _ckpt_meta(primary)
            trained_at = datetime.fromtimestamp(primary.stat().st_mtime)
            runs.append({
                "name": d.name,
                "path": str(primary.resolve()),
                "which": primary.name,
                "has_best": best.exists(),
                "has_last": last.exists(),
                "trained_at": trained_at.isoformat(timespec="seconds"),
                "trained_at_label": trained_at.strftime("%d %b %Y, %H:%M"),
                "best_val": summary.get("best_val", meta.get("best_val")),
                "final_val": summary.get("val"),
                "final_train": summary.get("train"),
                "iters": summary.get("iter", meta.get("iter")),
                "minutes": summary.get("minutes"),
                "params": meta.get("params"),
                "config": meta.get("config", {}),
                "device": meta.get("device"),
                "error": meta.get("error"),
                "curve": _curve(d / "log.csv"),
                "active": str(primary.resolve()) == active,
            })
    # newest first
    runs.sort(key=lambda r: r["trained_at"], reverse=True)
    return {"runs": runs, "active": active, "root": str(CKPT_ROOT)}


def resolve_run_path(raw: str) -> Path:
    """Validate a checkpoint path from the browser: it must be an existing .pt inside
    llm/checkpoints/ (never an arbitrary file elsewhere on disk)."""
    p = Path(raw).resolve()
    root = CKPT_ROOT.resolve()
    if root not in p.parents:
        raise ValueError("checkpoint must be inside llm/checkpoints/")
    if p.suffix != ".pt" or not p.exists():
        raise ValueError("not an existing .pt checkpoint")
    return p
