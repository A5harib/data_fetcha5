"""
Direction: a stretch / snap-back score that makes a call only when it is extreme.

Research on 2023-2026 walk-forward folds found the usable signal is mean
reversion: when price is stretched from its recent VWAP and range, the median
next move points back. Squared-loss models (ridge, trees) chase the mean move,
which has no stable sign, so they miss it. A rank-based score does not.

Fit (train rows only):
  1. winsorize each feature at its 0.5/99.5 percentiles, z-score
  2. keep the K features with the largest |Spearman IC| against ret_z
  3. score = mean(sign(IC) * z) over those K
  4. threshold = quantile `call_q` of |score|; below it the model abstains
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Calendar features carry no direction and only add noise to the selection.
EXCLUDE = {"hour_sin", "hour_cos", "weekend", "min_to_funding"}


@dataclass
class DirectionModel:
    features: list
    signs: list
    lo: list
    hi: list
    mean: list
    sd: list
    threshold: float

    @classmethod
    def fit(cls, X: pd.DataFrame, ret_z: pd.Series, k: int = 5, call_q: float = 0.8) -> "DirectionModel":
        cols = [c for c in X.columns if c not in EXCLUDE]
        A = X[cols]
        lo, hi = A.quantile(0.005), A.quantile(0.995)
        Z = A.clip(lo, hi, axis=1)
        mu, sd = Z.mean(), Z.std().replace(0, 1)
        Z = (Z - mu) / sd
        y = ret_z.clip(-6, 6).to_numpy()
        ic = {c: spearmanr(Z[c].to_numpy(), y)[0] for c in cols}
        top = sorted((c for c in cols if np.isfinite(ic[c])), key=lambda c: -abs(ic[c]))[:k]
        m = cls(top, [float(np.sign(ic[c])) for c in top], [float(lo[c]) for c in top],
                [float(hi[c]) for c in top], [float(mu[c]) for c in top], [float(sd[c]) for c in top], 0.0)
        m.threshold = float(np.quantile(np.abs(m.score(X)), call_q))
        return m

    def score(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.features].to_numpy(float)
        Z = (np.clip(A, self.lo, self.hi) - np.asarray(self.mean)) / np.asarray(self.sd)
        return (Z * np.asarray(self.signs)).mean(axis=1)

    def call(self, X: pd.DataFrame) -> np.ndarray:
        """+1 up, -1 down, 0 no call."""
        s = self.score(X)
        return np.where(np.abs(s) > self.threshold, np.sign(s), 0.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "DirectionModel":
        return cls(**json.loads(s))
