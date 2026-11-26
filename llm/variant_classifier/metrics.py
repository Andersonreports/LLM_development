"""Ordinal metrics. The headline number is quadratic-weighted kappa (QWK): it rewards
getting the tier right and penalises being off by two tiers four times as much as being
off by one, which is exactly what we want clinically."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .schema import N_TIERS, TIERS

# "dangerous" = the model and the truth disagree across the benign/pathogenic divide.
_BENIGN = {0, 1}        # B, LB
_PATHOGENIC = {3, 4}    # LP, P


def evaluate(y_true, y_pred_tier, score=None) -> dict:
    """y_true / y_pred_tier are integer tiers 0..4. `score` is an optional continuous
    expected-tier used for Spearman (falls back to the predicted tier)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_tier, dtype=int)
    s = np.asarray(score if score is not None else y_pred, dtype=float)

    labels = list(range(N_TIERS))
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)
    mae = float(np.abs(y_true - y_pred).mean())
    rho = float(spearmanr(s, y_true).statistic) if len(set(y_true)) > 1 else float("nan")
    exact = float((y_true == y_pred).mean())
    within1 = float((np.abs(y_true - y_pred) <= 1).mean())

    dangerous = int(sum((t in _BENIGN and p in _PATHOGENIC) or
                        (t in _PATHOGENIC and p in _BENIGN)
                        for t, p in zip(y_true, y_pred)))
    return {
        "qwk": round(qwk, 4),
        "mae_tiers": round(mae, 4),
        "spearman": round(rho, 4),
        "exact_acc": round(exact, 4),
        "within_1_tier": round(within1, 4),
        "dangerous_errors": dangerous,
        "dangerous_rate": round(dangerous / max(1, len(y_true)), 5),
        "n": int(len(y_true)),
    }


def confusion(y_true, y_pred_tier) -> str:
    cm = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred_tier, int),
                          labels=list(range(N_TIERS)))
    head = "true\\pred " + " ".join(f"{t:>5}" for t in TIERS)
    rows = [head]
    for i, t in enumerate(TIERS):
        rows.append(f"{t:>8}  " + " ".join(f"{cm[i, j]:>5}" for j in range(N_TIERS)))
    return "\n".join(rows)


def baseline_from_score(y_true, raw_score) -> tuple[np.ndarray, np.ndarray]:
    """A single-feature baseline (e.g. REVEL): rank variants by the score and cut into tiers
    matching the observed tier frequencies, so it competes on equal footing with the model.
    Returns (predicted_tiers, continuous_score). NaNs rank lowest (least pathogenic)."""
    y_true = np.asarray(y_true, dtype=int)
    s = np.asarray(raw_score, dtype=float)
    order_val = np.where(np.isnan(s), -np.inf, s)
    ranks = order_val.argsort().argsort()                # 0 = lowest score
    frac = ranks / max(1, len(ranks) - 1)
    # tier cut points from the true tier proportions (cumulative)
    counts = np.bincount(y_true, minlength=N_TIERS)
    cum = np.cumsum(counts) / counts.sum()
    pred = np.searchsorted(cum[:-1], frac, side="right")
    return pred.astype(int), frac * (N_TIERS - 1)
