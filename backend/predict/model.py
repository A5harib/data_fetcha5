"""
The short-horizon predictor and its retrain rules.

Design decisions worth stating up front, because they are the fixes for how
the old reversal model went wrong:

- Regression, not classification. Predicting the size of the next move keeps
  information that a yes/no label throws away.
- The scaler is fit ONCE and frozen into the artifact. Refitting on every
  retrain silently moves the meaning of the output, so a threshold tuned on
  Monday means something else by Friday.
- Sample weights come from wall-clock age, not row position.
- The historical base always stays in the training mix. Training on only the
  last few hours produces a model that forgets.
- A retrained candidate must BEAT the incumbent on held-out recent data
  before it is allowed to take over. Otherwise one bad batch destroys a
  working model, permanently, because the old code also overwrote the file.
- Saving is write-temp-then-rename, so a crash cannot leave a corrupt model.
- Feature order is stored in the artifact and checked on load.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from predict.features import FEATURE_NAMES

logger = logging.getLogger("predict.model")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_HORIZON_SEC = 30
# Half-life for recency weighting. A sample 6h old counts half as much as a
# fresh one. Long enough to keep a day of context, short enough to adapt.
RECENCY_HALFLIFE_SEC = 6 * 3600
# A candidate must beat the incumbent by this much on the holdout to swap in.
# Requiring a clear win, not a tie, stops noise from churning the model.
SWAP_MARGIN = 0.002


@dataclass
class ModelMeta:
    feature_names: List[str]
    horizon_sec: int
    trained_at: float
    n_train_samples: int
    holdout_ic: float
    scaler_mean: List[float]
    scaler_std: List[float]
    model_kind: str

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "horizon_sec": self.horizon_sec,
            "trained_at": self.trained_at,
            "n_train_samples": self.n_train_samples,
            "holdout_ic": self.holdout_ic,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "model_kind": self.model_kind,
        }

    @staticmethod
    def from_dict(d: dict) -> "ModelMeta":
        return ModelMeta(
            feature_names=d["feature_names"],
            horizon_sec=d["horizon_sec"],
            trained_at=d["trained_at"],
            n_train_samples=d["n_train_samples"],
            holdout_ic=d["holdout_ic"],
            scaler_mean=d["scaler_mean"],
            scaler_std=d["scaler_std"],
            model_kind=d.get("model_kind", "unknown"),
        )


class FrozenScaler:
    """
    Standardization with fixed parameters.

    Deliberately has no refit path. The whole point is that the numbers going
    into the model mean the same thing next week as they do today, so the
    only way to change them is to train a new model artifact.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        # Guard against a constant feature producing division by zero.
        self.std = np.where(np.asarray(std, dtype=np.float64) < 1e-9, 1.0, np.asarray(std, dtype=np.float64))

    @staticmethod
    def fit(X: np.ndarray) -> "FrozenScaler":
        X = np.asarray(X, dtype=np.float64)
        return FrozenScaler(mean=X.mean(axis=0), std=X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std


def recency_weights(timestamps_ms: np.ndarray, now_ms: Optional[int] = None) -> np.ndarray:
    """
    Exponential decay on actual age.

    The old code used np.linspace over row position, which meant the weights
    landed on rows by their index rather than by how old they were. Mixing
    two data sources scrambled it completely.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    age_sec = np.maximum(0.0, (now_ms - np.asarray(timestamps_ms, dtype=np.float64)) / 1000.0)
    # 0.5 ** (age / halflife), not exp(-age / halflife). The latter decays to
    # 1/e at the halflife, which would make the constant's name a lie.
    return np.exp(-np.log(2.0) * age_sec / RECENCY_HALFLIFE_SEC)


class RidgeCore:
    """
    Ridge regression via numpy. No dependencies beyond numpy.

    Chosen as the default because at a 30s horizon the honest signal is weak
    and close to linear. A gradient-boosted model on 15 features and a
    near-zero edge mostly memorizes noise. GBMCore below is available when
    there is enough recorded data to justify it.
    """

    kind = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        w = w / (w.sum() + 1e-12) * n

        # Weighted centering, so the intercept is not folded into the penalty.
        wx_mean = (w[:, None] * X).sum(axis=0) / w.sum()
        wy_mean = float((w * y).sum() / w.sum())
        Xc = X - wx_mean
        yc = y - wy_mean

        sw = np.sqrt(w)[:, None]
        Xw = Xc * sw
        yw = yc * np.sqrt(w)

        A = Xw.T @ Xw + self.alpha * np.eye(d)
        b = Xw.T @ yw
        self.coef = np.linalg.solve(A, b)
        self.intercept = wy_mean - float(wx_mean @ self.coef)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(X, dtype=np.float64) @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "alpha": self.alpha,
            "coef": self.coef.tolist() if self.coef is not None else None,
            "intercept": self.intercept,
        }

    @staticmethod
    def from_dict(d: dict) -> "RidgeCore":
        m = RidgeCore(alpha=d.get("alpha", 1.0))
        m.coef = np.asarray(d["coef"], dtype=np.float64) if d.get("coef") is not None else None
        m.intercept = float(d.get("intercept", 0.0))
        return m


class GBMCore:
    """
    Gradient boosting, for when there is enough recorded data to support it.

    Requires xgboost. Kept behind an explicit opt-in because more capacity
    against a weak signal usually means more overfitting, not more edge.
    """

    kind = "gbm"

    def __init__(self, **params):
        self.params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "random_state": 42,
            **params,
        }
        self.model = None

    def fit(self, X, y, sample_weight=None):
        import xgboost as xgb

        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(np.asarray(X), np.asarray(y), sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=np.float64)
        return np.asarray(self.model.predict(np.asarray(X)), dtype=np.float64)

    def to_dict(self) -> dict:
        import base64
        import tempfile

        if self.model is None:
            return {"kind": self.kind, "params": self.params, "booster": None}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp = tf.name
        try:
            self.model.save_model(tmp)
            raw = Path(tmp).read_bytes()
        finally:
            os.unlink(tmp)
        return {
            "kind": self.kind,
            "params": self.params,
            "booster": base64.b64encode(raw).decode("ascii"),
        }

    @staticmethod
    def from_dict(d: dict) -> "GBMCore":
        import base64
        import tempfile

        import xgboost as xgb

        m = GBMCore(**d.get("params", {}))
        if d.get("booster"):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tf.write(base64.b64decode(d["booster"]))
                tmp = tf.name
            try:
                m.model = xgb.XGBRegressor(**m.params)
                m.model.load_model(tmp)
            finally:
                os.unlink(tmp)
        return m


def _make_core(kind: str):
    if kind == "gbm":
        return GBMCore()
    return RidgeCore()


class Predictor:
    """
    Loads, serves, and safely replaces a short-horizon model.

    predict() returns a dict the API and UI can render directly, including an
    explicit "no opinion" state. A model that cannot say "I don't know" will
    happily emit a confident number from garbage input.
    """

    def __init__(self, symbol: str = "BTCUSDT", model_dir: Optional[Path] = None):
        self.symbol = symbol.upper()
        self.model_dir = model_dir or MODEL_DIR
        self.core = None
        self.scaler: Optional[FrozenScaler] = None
        self.meta: Optional[ModelMeta] = None
        self.swap_count = 0
        self.rejected_count = 0

    @property
    def path(self) -> Path:
        return self.model_dir / f"{self.symbol}_short_horizon.json"

    def is_ready(self) -> bool:
        return self.core is not None and self.scaler is not None and self.meta is not None

    def load(self) -> bool:
        """
        Load from disk. Returns False rather than raising, so a missing model
        is a normal cold-start state and not a crash. Never falls back to
        training on synthetic data, which is what the old predictor did.
        """
        if not self.path.exists():
            logger.info(f"No model at {self.path}. Predictor stays unready until trained.")
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            meta = ModelMeta.from_dict(payload["meta"])
            if meta.feature_names != FEATURE_NAMES:
                logger.error(
                    "Feature order in the saved model does not match the current code. "
                    "Refusing to load, because this would silently feed the wrong "
                    "number into every slot. Retrain."
                )
                return False
            self.meta = meta
            self.scaler = FrozenScaler(np.array(meta.scaler_mean), np.array(meta.scaler_std))
            self.core = (
                GBMCore.from_dict(payload["core"])
                if payload["core"].get("kind") == "gbm"
                else RidgeCore.from_dict(payload["core"])
            )
            logger.info(
                f"Loaded {meta.model_kind} model for {self.symbol}: "
                f"horizon {meta.horizon_sec}s, holdout IC {meta.holdout_ic:.4f}, "
                f"{meta.n_train_samples} training samples"
            )
            return True
        except Exception as e:
            logger.error(f"Could not load model at {self.path}: {e}")
            return False

    def save(self):
        """Write to a temp file then rename, so an interrupted save cannot corrupt the model."""
        if not self.is_ready():
            raise RuntimeError("Refusing to save an unready predictor.")
        payload = {"meta": self.meta.to_dict(), "core": self.core.to_dict()}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)
        logger.info(f"Saved model -> {self.path}")

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self.is_ready():
            return np.zeros(len(X), dtype=np.float64)
        return self.core.predict(self.scaler.transform(X))

    def predict(self, feature_dict: Dict[str, float]) -> dict:
        """
        One live prediction.

        Output is in units of 60s realized volatility, because that is what
        the model was trained against. expected_move_pct converts it back to
        a percentage using the current vol, which is the number a human can
        actually read.
        """
        if not self.is_ready():
            return {
                "ready": False,
                "direction": "NO_MODEL",
                "score": 0.0,
                "expected_move_pct": 0.0,
                "confidence": 0.0,
                "horizon_sec": DEFAULT_HORIZON_SEC,
            }

        missing = [n for n in FEATURE_NAMES if n not in feature_dict]
        if missing:
            # Guessing a value for a missing feature is how a model ends up
            # confidently wrong. Say so instead.
            return {
                "ready": False,
                "direction": "INSUFFICIENT_DATA",
                "score": 0.0,
                "expected_move_pct": 0.0,
                "confidence": 0.0,
                "horizon_sec": self.meta.horizon_sec,
                "missing_features": missing,
            }

        x = np.array([[float(feature_dict[n]) for n in FEATURE_NAMES]], dtype=np.float64)
        score = float(self.predict_raw(x)[0])

        vol = max(float(feature_dict.get("realized_vol_60s", 0.0)), 1e-6)
        expected_move_pct = score * vol * 100.0

        # Confidence is bounded and deliberately modest. A 30s edge is small;
        # a number that reads like 95% certainty would be a lie.
        confidence = float(np.clip(abs(score) / 2.0, 0.0, 1.0))

        if abs(score) < 0.15:
            direction = "FLAT"
        elif score > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        return {
            "ready": True,
            "direction": direction,
            "score": round(score, 4),
            "expected_move_pct": round(expected_move_pct, 4),
            "confidence": round(confidence, 3),
            "horizon_sec": self.meta.horizon_sec,
            "model_ic": round(self.meta.holdout_ic, 4),
        }

    def try_swap(
        self,
        candidate_core,
        candidate_scaler: FrozenScaler,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        n_train: int,
        horizon_sec: int,
    ) -> bool:
        """
        Replace the live model only if the candidate genuinely scores better
        on recent held-out data.

        This is the guard the old continual learner lacked. It hot-swapped
        whatever came out of training and wrote it straight to disk, so a
        single bad batch replaced a working model with no way back.
        """
        from predict.evaluate import information_coefficient

        if len(y_holdout) < 50:
            logger.warning(f"Holdout of {len(y_holdout)} is too small to judge a swap. Rejecting.")
            self.rejected_count += 1
            return False

        cand_ic = information_coefficient(candidate_core.predict(candidate_scaler.transform(X_holdout)), y_holdout)

        if self.is_ready():
            incumbent_ic = information_coefficient(self.predict_raw(X_holdout), y_holdout)
        else:
            # Cold start: accept anything that shows a positive signal.
            incumbent_ic = 0.0

        if cand_ic <= incumbent_ic + SWAP_MARGIN:
            logger.info(
                f"Rejected candidate: IC {cand_ic:.4f} does not beat incumbent "
                f"{incumbent_ic:.4f} by the {SWAP_MARGIN} margin. Keeping current model."
            )
            self.rejected_count += 1
            return False

        self.core = candidate_core
        self.scaler = candidate_scaler
        self.meta = ModelMeta(
            feature_names=list(FEATURE_NAMES),
            horizon_sec=horizon_sec,
            trained_at=time.time(),
            n_train_samples=n_train,
            holdout_ic=cand_ic,
            scaler_mean=candidate_scaler.mean.tolist(),
            scaler_std=candidate_scaler.std.tolist(),
            model_kind=candidate_core.kind,
        )
        self.swap_count += 1
        self.save()
        logger.info(f"Swapped in new model: IC {incumbent_ic:.4f} -> {cand_ic:.4f} (swap #{self.swap_count})")
        return True

    def status(self) -> dict:
        return {
            "symbol": self.symbol,
            "ready": self.is_ready(),
            "swaps_accepted": self.swap_count,
            "swaps_rejected": self.rejected_count,
            "horizon_sec": self.meta.horizon_sec if self.meta else DEFAULT_HORIZON_SEC,
            "holdout_ic": round(self.meta.holdout_ic, 4) if self.meta else None,
            "trained_at": self.meta.trained_at if self.meta else None,
            "n_train_samples": self.meta.n_train_samples if self.meta else 0,
        }


def demo():
    """Self-check for the parts that would fail silently: weights, frozen scaler, swap guard, round-trip."""
    print("model.py self-check")
    rng = np.random.default_rng(0)

    # Recency weights follow clock age, not row order.
    now = 1_700_000_000_000
    ts = np.array([now - 12 * 3600 * 1000, now - 6 * 3600 * 1000, now], dtype=np.int64)
    w = recency_weights(ts, now_ms=now)
    assert w[2] > w[1] > w[0], f"weights not decreasing with age: {w}"
    assert abs(w[1] - 0.5) < 0.01, f"6h half-life wrong: {w[1]:.4f}"
    # Shuffling row order must not change any row's weight.
    order = np.array([2, 0, 1])
    w2 = recency_weights(ts[order], now_ms=now)
    assert np.allclose(w2, w[order]), "weights depend on position, not age"
    print(f"  recency weights track age, not row order (6h -> {w[1]:.3f})")

    # A frozen scaler must not drift when new data arrives.
    X1 = rng.normal(loc=5.0, scale=2.0, size=(500, len(FEATURE_NAMES)))
    sc = FrozenScaler.fit(X1)
    before = sc.transform(X1[:5]).copy()
    X2 = rng.normal(loc=50.0, scale=20.0, size=(500, len(FEATURE_NAMES)))
    _ = sc.transform(X2)
    after = sc.transform(X1[:5])
    assert np.allclose(before, after), "scaler output changed after seeing new data"
    assert not hasattr(sc, "fit_transform"), "frozen scaler exposes a refit path"
    print("  scaler frozen: same input maps to the same output regardless of later data")

    # Ridge recovers a planted linear relationship.
    Xr = rng.normal(size=(2000, 3))
    yr = 0.7 * Xr[:, 0] - 0.4 * Xr[:, 1] + rng.normal(scale=0.3, size=2000)
    core = RidgeCore(alpha=1.0).fit(Xr, yr)
    assert abs(core.coef[0] - 0.7) < 0.1 and abs(core.coef[1] + 0.4) < 0.1, f"ridge off: {core.coef}"
    print(f"  ridge recovers planted coefficients {np.round(core.coef, 3)}")

    # Weighted fit actually respects the weights.
    yw = yr.copy()
    yw[:1000] = -yw[:1000]
    heavy = np.concatenate([np.full(1000, 0.01), np.full(1000, 1.0)])
    cw = RidgeCore(alpha=1.0).fit(Xr, yw, sample_weight=heavy)
    assert cw.coef[0] > 0.3, f"weights ignored: {cw.coef}"
    print("  sample weights change the fit as expected")

    # Swap guard: a worse candidate must be rejected.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Predictor(symbol="TEST", model_dir=Path(td))
        Xh = rng.normal(size=(400, len(FEATURE_NAMES)))
        good_w = rng.normal(size=len(FEATURE_NAMES))
        yh = Xh @ good_w + rng.normal(scale=0.5, size=400)
        sc2 = FrozenScaler.fit(Xh)

        good = RidgeCore().fit(sc2.transform(Xh), yh)
        assert p.try_swap(good, sc2, Xh, yh, n_train=400, horizon_sec=30), "cold start rejected a good model"
        assert p.is_ready() and p.path.exists()

        junk = RidgeCore()
        junk.coef = rng.normal(size=len(FEATURE_NAMES)) * 0.001
        junk.intercept = 0.0
        assert not p.try_swap(junk, sc2, Xh, yh, n_train=400, horizon_sec=30), "junk candidate was accepted"
        assert p.swap_count == 1 and p.rejected_count == 1
        print(f"  swap guard: accepted {p.swap_count}, rejected {p.rejected_count}")

        # Round-trip through disk must reproduce predictions exactly.
        before_pred = p.predict_raw(Xh[:20]).copy()
        p2 = Predictor(symbol="TEST", model_dir=Path(td))
        assert p2.load(), "failed to reload saved model"
        assert np.allclose(before_pred, p2.predict_raw(Xh[:20])), "predictions changed across save/load"
        print("  save/load round-trip preserves predictions exactly")

        # Missing features must produce an explicit refusal, not a number.
        out = p2.predict({"book_imbalance": 0.1})
        assert out["ready"] is False and out["direction"] == "INSUFFICIENT_DATA", out
        print("  incomplete input returns INSUFFICIENT_DATA instead of a guess")

    # A missing model file is a normal cold-start, not a crash or a synthetic fallback.
    with tempfile.TemporaryDirectory() as td:
        p3 = Predictor(symbol="NOPE", model_dir=Path(td))
        assert p3.load() is False and p3.is_ready() is False
        assert p3.predict({n: 0.0 for n in FEATURE_NAMES})["direction"] == "NO_MODEL"
        print("  missing model reports NO_MODEL, never trains on synthetic data")

    print("all checks passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    demo()
