"""Score checkpoints on the whole validation and test text (not random samples), so
results are exact and comparable between experiments.

    python evaluate.py                                  # every checkpoints/*/best.pt
    python evaluate.py --ckpt checkpoints/exp_b/best.pt --split val test

Reports loss and perplexity (e^loss). Use --split test only for final comparisons:
choosing settings by test loss would turn the test set into a second validation set.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from generate import load

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def full_loss(model, tokens: np.ndarray, device: str, batch_size: int = 32) -> float:
    """Average next-token loss over every token, using non-overlapping windows."""
    T = model.cfg.block_size
    n = (len(tokens) - 1) // T  # whole windows only
    x_all = torch.from_numpy(tokens[:n * T].astype(np.int64)).view(n, T)
    y_all = torch.from_numpy(tokens[1:n * T + 1].astype(np.int64)).view(n, T)
    total = 0.0
    for i in range(0, n, batch_size):
        x, y = x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)
        _, loss = model(x, y)
        total += loss.item() * len(x)
    return total / n


def evaluate(ckpt: Path, splits: list[str], device: str) -> dict:
    model, _ = load(ckpt, device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    data = Path(ck["train_args"]["data"])
    out = {"checkpoint": str(ckpt), "iter": ck["iter"], "params_M": round(model.num_params() / 1e6, 2)}
    for s in splits:
        loss = full_loss(model, np.memmap(data / f"{s}.bin", dtype=np.uint16, mode="r"), device)
        out[f"{s}_loss"], out[f"{s}_ppl"] = round(loss, 4), round(math.exp(loss), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, nargs="*")
    ap.add_argument("--split", nargs="+", default=["val"], choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    ckpts = a.ckpt or sorted((ROOT / "checkpoints").glob("*/best.pt"))
    for c in ckpts:
        print(json.dumps(evaluate(c, a.split, a.device)))


if __name__ == "__main__":
    main()
