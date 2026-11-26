"""Train and honestly evaluate the ordinal model.

    python -m variant_classifier.train --demo                 # synthetic data, gene split
    python -m variant_classifier.train --demo --split random  # show the leakage inflation
    python -m variant_classifier.train --table data.csv --save model.pkl

The evaluation uses out-of-fold predictions so every variant is scored by a model that did
NOT see it. --split gene (the default, GroupKFold on gene) is the honest setting; --split
random exists only to demonstrate how a careless split inflates the numbers."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.inspection import permutation_importance  # noqa: F401 (kept for reference)

from .metrics import baseline_from_score, confusion, evaluate
from .ordinal import OrdinalGBM
from .schema import FEATURES


def _xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(dtype=float), df["tier"].to_numpy(dtype=int)


def cross_val(df: pd.DataFrame, split: str = "gene", n_splits: int = 5,
              seed: int = 0) -> dict:
    X, y = _xy(df)
    groups = df["gene"].astype("category").cat.codes.to_numpy()
    n_oof = np.full(len(df), -1)
    oof_tier = np.zeros(len(df), dtype=int)
    oof_score = np.zeros(len(df), dtype=float)

    if split == "gene":
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    elif split == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    else:
        raise ValueError("split must be 'gene' or 'random'")

    for tr, va in splitter:
        model = OrdinalGBM().fit(X[tr], y[tr])
        oof_tier[va] = model.predict(X[va])
        oof_score[va] = model.expected_tier(X[va])
        n_oof[va] = 1

    assert (n_oof == 1).all(), "every row should be scored exactly once"
    model_metrics = evaluate(y, oof_tier, oof_score)

    # single-score baseline (REVEL) on the same variants
    base_tier, base_score = baseline_from_score(y, df["revel"].to_numpy(dtype=float))
    base_metrics = evaluate(y, base_tier, base_score)

    return {"split": split, "model": model_metrics, "baseline_revel": base_metrics,
            "confusion": confusion(y, oof_tier)}


def permutation_importance_qwk(df: pd.DataFrame, seed: int = 0) -> list[tuple[str, float]]:
    """How much QWK drops when each feature is shuffled (fit on 80%, score on 20%)."""
    X, y = _xy(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, va = idx[:cut], idx[cut:]
    model = OrdinalGBM().fit(X[tr], y[tr])
    base = evaluate(y[va], model.predict(X[va]))["qwk"]
    out = []
    for j, feat in enumerate(FEATURES):
        Xp = X[va].copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        drop = base - evaluate(y[va], model.predict(Xp))["qwk"]
        out.append((feat, round(drop, 4)))
    return sorted(out, key=lambda kv: kv[1], reverse=True)


def _report(res: dict) -> None:
    m, b = res["model"], res["baseline_revel"]
    print(f"\n=== ordinal model ({res['split']} split, out-of-fold) ===")
    print(f"  QWK {m['qwk']:.3f} | MAE {m['mae_tiers']:.3f} tiers | Spearman {m['spearman']:.3f} "
          f"| exact {m['exact_acc']:.3f} | within-1 {m['within_1_tier']:.3f}")
    print(f"  dangerous P/B flips: {m['dangerous_errors']} ({m['dangerous_rate']:.3%}) of {m['n']}")
    print(f"--- baseline: REVEL alone ---")
    print(f"  QWK {b['qwk']:.3f} | MAE {b['mae_tiers']:.3f} | Spearman {b['spearman']:.3f} "
          f"| dangerous {b['dangerous_errors']}")
    verdict = "beats" if m["qwk"] > b["qwk"] + 0.01 else ("ties" if m["qwk"] >= b["qwk"] - 0.01 else "LOSES to")
    print(f"  => model {verdict} the single-score baseline (QWK {m['qwk']:.3f} vs {b['qwk']:.3f})")
    print("\nconfusion (rows = truth, cols = prediction):")
    print(res["confusion"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", action="store_true", help="use synthetic data")
    src.add_argument("--table", type=Path, help="prepared .csv/.parquet (gene, tier, features)")
    ap.add_argument("--split", choices=["gene", "random"], default="gene")
    ap.add_argument("--importance", action="store_true", help="also show feature importance")
    ap.add_argument("--save", type=Path, help="train a final model on all rows and pickle it")
    a = ap.parse_args()

    if a.demo:
        from .synth import make
        df = make()
        print(f"synthetic: {len(df):,} variants, {df['gene'].nunique()} genes, "
              f"tier counts {df['tier'].value_counts().sort_index().to_dict()}")
    else:
        from .data import load_table
        df = load_table(a.table)
        print(f"loaded {len(df):,} variants from {a.table}")

    _report(cross_val(df, split=a.split))

    if a.importance:
        print("\nfeature importance (QWK drop when shuffled):")
        for feat, drop in permutation_importance_qwk(df):
            print(f"  {feat:22} {drop:+.4f}")

    if a.save:
        X, y = _xy(df)
        model = OrdinalGBM().fit(X, y)
        with open(a.save, "wb") as f:
            pickle.dump({"model": model, "features": FEATURES}, f)
        print(f"\nsaved final model -> {a.save}")


if __name__ == "__main__":
    main()
