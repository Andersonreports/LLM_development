"""End-to-end check of the training pipeline on a tiny CPU model: loss goes down,
checkpoints are written, resume continues from the saved iteration, generation loads it."""
from __future__ import annotations

import numpy as np
import torch

from generate import load
from tinygpt import CharTokenizer
from train import TrainArgs, lr_at, train

TEXT = "to be or not to be, that is the question. " * 400


def _prepare(tmp_path):
    tok = CharTokenizer.train(TEXT)
    tok.save(tmp_path / "tokenizer.json")
    ids = np.array(tok.encode(TEXT), dtype=np.uint16)
    ids[:-2000].tofile(tmp_path / "train.bin")
    ids[-2000:].tofile(tmp_path / "val.bin")


def _args(tmp_path, **kw) -> TrainArgs:
    return TrainArgs(**{**dict(data=str(tmp_path), name="t", preset="tiny", block_size=32, batch_size=8,
                               max_iters=60, eval_interval=30, eval_iters=5, warmup_iters=5, lr=3e-3,
                               sample_tokens=0, device="cpu", out_dir=str(tmp_path / "ck")), **kw})


def test_training_lowers_loss_and_resumes(tmp_path):
    _prepare(tmp_path)
    first = train(_args(tmp_path))
    log = (tmp_path / "ck" / "t" / "log.csv").read_text().splitlines()
    losses = [float(row.split(",")[2]) for row in log[1:]]
    assert losses[-1] < losses[0] - 0.5, losses          # validation loss dropped
    assert (tmp_path / "ck" / "t" / "best.pt").exists()

    resumed = train(_args(tmp_path, max_iters=90, resume=True))
    assert resumed["iter"] == 90
    ck = torch.load(tmp_path / "ck" / "t" / "last.pt", weights_only=False)
    assert ck["iter"] == 90 and ck["best_val"] <= first["best_val"]

    model, tok = load(tmp_path / "ck" / "t" / "best.pt", "cpu")
    out = model.generate(torch.tensor([tok.encode("to ")]), 20, temperature=0)
    assert len(tok.decode(out[0].tolist())) == 23


def test_size_overrides_and_full_evaluation(tmp_path):
    from evaluate import evaluate
    _prepare(tmp_path)
    np.fromfile(tmp_path / "val.bin", dtype=np.uint16).tofile(tmp_path / "test.bin")
    train(_args(tmp_path, max_iters=30, n_layer=1, n_embd=64, n_head=2))
    ck = torch.load(tmp_path / "ck" / "t" / "best.pt", weights_only=False)
    assert (ck["config"]["n_layer"], ck["config"]["n_embd"]) == (1, 64)
    res = evaluate(tmp_path / "ck" / "t" / "best.pt", ["val", "test"], "cpu")
    assert res["val_loss"] == res["test_loss"] > 0      # identical text -> identical deterministic score
    assert abs(res["val_ppl"] - np.exp(res["val_loss"])) < 0.01 * res["val_ppl"]


def test_learning_rate_schedule():
    a = TrainArgs(lr=1e-3, min_lr=1e-4, warmup_iters=10, max_iters=110)
    assert lr_at(0, a) == 1e-4                            # warm-up starts small
    assert abs(lr_at(9, a) - 1e-3) < 1e-12               # reaches the peak
    assert abs(lr_at(110, a) - 1e-4) < 1e-12             # decays to min_lr
