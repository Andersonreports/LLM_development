"""An ordinal classifier for the 5 ACMG tiers, using the Frank & Hall decomposition.

Idea: an ordered target y in {0..K-1} is turned into K-1 binary questions
    is y > 0?   is y > 1?   ...   is y > K-2?
Each is answered by its own gradient-boosted tree model. The per-tier probability is then
the difference of neighbouring cumulative probabilities:
    P(y=0)   = 1 - P(y>0)
    P(y=k)   = P(y>k-1) - P(y>k)
    P(y=K-1) = P(y>K-2)
Negatives from non-monotone estimates are clipped and the row renormalised. This keeps the
ordering information that a plain multiclass model throws away, and still gives a full tier
distribution (so we can report an expected-tier score as well as the argmax tier)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import N_TIERS


def _base_model(**kw) -> HistGradientBoostingClassifier:
    # HistGradientBoosting handles NaN features natively (many variants lack some scores).
    params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  l2_regularization=1.0, early_stopping=True, random_state=0)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


class OrdinalGBM:
    def __init__(self, n_tiers: int = N_TIERS, **model_kw):
        self.n_tiers = n_tiers
        self.model_kw = model_kw
        self.models_: list[HistGradientBoostingClassifier | None] = []
        self.constant_: list[int | None] = []  # fold where one side is empty

    def fit(self, X, y) -> "OrdinalGBM":
        y = np.asarray(y)
        self.models_, self.constant_ = [], []
        for k in range(self.n_tiers - 1):
            target = (y > k).astype(int)
            if target.min() == target.max():
                # all on one side of the threshold in this (sub)sample
                self.models_.append(None)
                self.constant_.append(int(target[0]))
            else:
                self.models_.append(_base_model(**self.model_kw).fit(X, target))
                self.constant_.append(None)
        return self

    def _cum_gt(self, X) -> np.ndarray:
        """P(y > k) for k = 0..K-2, shape (n, K-1)."""
        n = X.shape[0]
        cols = []
        for model, const in zip(self.models_, self.constant_):
            if model is None:
                cols.append(np.full(n, float(const)))
            else:
                cols.append(model.predict_proba(X)[:, 1])
        return np.vstack(cols).T if cols else np.zeros((n, 0))

    def predict_proba(self, X) -> np.ndarray:
        """Per-tier probability, shape (n, K)."""
        gt = self._cum_gt(X)
        n = X.shape[0]
        p = np.zeros((n, self.n_tiers))
        prev = np.ones(n)                       # P(y > -1) = 1
        for k in range(self.n_tiers - 1):
            p[:, k] = np.clip(prev - gt[:, k], 0, None)
            prev = gt[:, k]
        p[:, -1] = np.clip(prev, 0, None)
        s = p.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return p / s

    def predict(self, X) -> np.ndarray:
        """Most-likely tier (argmax)."""
        return self.predict_proba(X).argmax(axis=1)

    def expected_tier(self, X) -> np.ndarray:
        """Continuous score in [0, K-1] = sum_k k * P(y=k); good for ranking and MAE."""
        p = self.predict_proba(X)
        return p @ np.arange(self.n_tiers)
