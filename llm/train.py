"""Train tinygpt on a prepared dataset (see prepare_data.py).

    python train.py                                   # small model on Tiny Shakespeare
    python train.py --preset tiny --max-iters 500     # quick experiment
    python train.py --resume                          # continue from the last checkpoint

Every --eval-interval steps it estimates training and validation loss, writes a line to
log.csv, prints a short text sample, and saves checkpoints:
  checkpoints/<name>/last.pt   always the latest state (used by --resume)
  checkpoints/<name>/best.pt   the state with the lowest validation loss so far
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tinygpt import GPT, load_tokenizer, preset

ROOT = Path(__file__).resolve().parent


@dataclass
class TrainArgs:
    data: str = str(ROOT / "data" / "shakespeare")
    name: str = "shakespeare_char"
    preset: str = "small"
    block_size: int | None = None      # override the preset's context window
    n_layer: int | None = None         # override the preset's number of blocks
    n_head: int | None = None          # override the preset's attention heads
    n_embd: int | None = None          # override the preset's width
    dropout: float = 0.2
    batch_size: int = 64
    max_iters: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100
    sample_tokens: int = 200
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: bool = False
    out_dir: str = str(ROOT / "checkpoints")


def lr_at(it: int, a: TrainArgs) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr."""
    if it < a.warmup_iters:
        return a.lr * (it + 1) / a.warmup_iters
    progress = min(1.0, (it - a.warmup_iters) / max(1, a.max_iters - a.warmup_iters))
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (a.lr - a.min_lr)


class Batches:
    """Random (input, target) windows from a token file; target = input shifted by one."""

    def __init__(self, path: Path, block_size: int, batch_size: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path.name} has only {len(self.data)} tokens; need more than block_size+1")
        self.block, self.batch, self.device = block_size, batch_size, device

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self.data) - self.block - 1, self.batch)
        x = torch.from_numpy(np.stack([self.data[i:i + self.block] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([self.data[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64))
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x.to(self.device), y.to(self.device)


def _autocast(device: str):
    # bf16 halves memory traffic on the RTX 3050 without the loss-scaling fp16 needs
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device, enabled=False)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, Batches], iters: int, device: str) -> dict[str, float]:
    model.eval()
    out = {}
    for name, batches in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = batches.get()
            with _autocast(device):
                losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(a: TrainArgs) -> dict:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    data = Path(a.data)
    tok = load_tokenizer(data / "tokenizer.json")
    ckpt_dir = Path(a.out_dir) / a.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"vocab_size": tok.vocab_size, "dropout": a.dropout}
    overrides |= {k: getattr(a, k) for k in ("block_size", "n_layer", "n_head", "n_embd") if getattr(a, k)}
    cfg = preset(a.preset, **overrides)
    model = GPT(cfg).to(a.device)
    opt = model.configure_optimizer(lr=a.lr, weight_decay=a.weight_decay)
    start_iter, best_val = 0, float("inf")

    if a.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        print(f"resumed from iteration {ck['iter']} (best val loss {best_val:.4f})")

    splits = {"train": Batches(data / "train.bin", cfg.block_size, a.batch_size, a.device),
              "val": Batches(data / "val.bin", cfg.block_size, a.batch_size, a.device)}
    print(f"model: {model.num_params() / 1e6:.2f}M parameters | vocab {cfg.vocab_size} | "
          f"context {cfg.block_size} | device {a.device}")
    print(f"initial loss should be about ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}")

    log_path = ckpt_dir / "log.csv"
    new_log = not (a.resume and log_path.exists())
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s", "gpu_mem_gb"])

    def save(path: Path, it: int) -> None:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
                    "iter": it, "best_val": best_val, "train_args": asdict(a),
                    "tokenizer": str((data / "tokenizer.json").resolve())}, path)

    t0 = time.time()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = {}
    for it in range(start_iter, a.max_iters + 1):
        lr = lr_at(it, a)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % a.eval_interval == 0 or it == a.max_iters:
            losses = estimate_loss(model, splits, a.eval_iters, a.device)
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else 0.0
            log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", f"{elapsed:.0f}", f"{mem:.2f}"])
            log_f.flush()
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(ckpt_dir / "best.pt", it)
                marker = "  * best so far, saved"
            save(ckpt_dir / "last.pt", it)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.1e} | "
                  f"{elapsed / 60:5.1f} min | GPU {mem:.2f} GB{marker}")
            if a.sample_tokens:
                model.eval()
                start = torch.tensor([tok.encode("\n")], device=a.device)
                sample = tok.decode(model.generate(start, a.sample_tokens, temperature=0.8, top_k=40)[0].tolist())
                model.train()
                print("   sample: " + sample.strip().replace("\n", "\n           ")[:400])
            result = {"iter": it, **losses, "best_val": best_val, "minutes": elapsed / 60}
            if it == a.max_iters:
                break

        x, y = splits["train"].get()
        with _autocast(a.device):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)  # avoid rare huge updates
        opt.step()

    log_f.close()
    (ckpt_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"done: best validation loss {best_val:.4f}; checkpoints in {ckpt_dir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, default in TrainArgs.__dataclass_fields__.items():
        val = default.default
        flag = "--" + f.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(val) if val is not None else int, default=val)
    train(TrainArgs(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
