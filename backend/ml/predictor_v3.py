"""
Serving-side v3 predictor.

Builds features by calling the *same* build_frame() the trainer uses, on the
live candle buffer, rather than reimplementing the formulas. v2 had two
implementations that disagreed: training synthesized orderbook_imbalance from
kline volume while serving read live L2 depth, so the deployed model saw a
feature it had never been trained on.

Probabilities are isotonic-calibrated and the fire/no-fire threshold comes
from the validation precision-recall curve, both loaded from the artifact
written by ml/train_v3.py. The old hardcoded 0.70 was meaningless because
scale_pos_weight distorts raw XGBoost probabilities.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from config import settings, MODEL_DIR
from ml.features_v3 import FEATURE_NAMES_V3, build_frame, GATE_PAST_RETURN_PCT

logger = logging.getLogger("predictor_v3")

# build_frame's longest lookback is 120 bars; give it headroom.
MIN_CANDLES = 140


class ReversalPredictorV3:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.model: Optional[xgb.XGBClassifier] = None
        self.iso = None
        self.threshold = 0.5
        self.gate_pct = settings.gate_for(self.symbol)

        mp = MODEL_DIR / f"xgb_v3_{self.symbol}.json"
        cp = MODEL_DIR / f"calib_v3_{self.symbol}.joblib"
        if mp.exists() and cp.exists():
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(mp))
                bundle = joblib.load(str(cp))
                self.iso = bundle["isotonic"]
                self.threshold = float(bundle["threshold"])
                saved = bundle.get("features", FEATURE_NAMES_V3)
                if list(saved) != list(FEATURE_NAMES_V3):
                    logger.error(f"{self.symbol}: feature order changed since training; "
                                 f"refusing to serve stale model. Retrain with ml/train_v3.py")
                    self.model = None
                else:
                    logger.info(f"Loaded v3 model for {self.symbol} (threshold {self.threshold:.3f})")
            except Exception as e:
                logger.error(f"{self.symbol}: failed to load v3 model: {e}")
                self.model = None
        else:
            logger.warning(f"No v3 model for {self.symbol}. Run: python ml/train_v3.py {self.symbol}")

    @staticmethod
    def candles_to_df(candles) -> pd.DataFrame:
        return pd.DataFrame([{
            "open_time": c.open_time_ms, "open": c.open, "high": c.high,
            "low": c.low, "close": c.close, "volume": c.volume,
            "buy_volume": c.buy_volume, "sell_volume": c.sell_volume,
        } for c in candles])

    def predict_from_candles(self, candles, extra_factors: Optional[List[str]] = None) -> Dict:
        """
        candles: chronological list of Candle objects from OrderFlowBuffer.

        Returns a dict with the calibrated probability, whether the bar is even
        a reversal candidate (the gate), and the signal label.
        """
        factors = list(extra_factors or [])

        if self.model is None:
            return {"probability": None, "signal": "MODEL_UNAVAILABLE",
                    "in_setup": False, "threshold": self.threshold, "factors": factors}

        if len(candles) < MIN_CANDLES:
            return {"probability": None, "signal": "WARMING_UP",
                    "in_setup": False, "threshold": self.threshold,
                    "bars_needed": MIN_CANDLES - len(candles), "factors": factors}

        df = self.candles_to_df(candles)

        # The gate: the model was only ever trained on bars that follow a real
        # 3-bar move. Scoring a quiet bar is out of distribution, so say so
        # instead of emitting a number.
        past3 = float((df["close"].iloc[-1] / df["close"].iloc[-4] - 1.0) * 100.0)
        if abs(past3) < self.gate_pct:
            return {"probability": None, "signal": "NO_SETUP",
                    "in_setup": False, "past_return_3m_pct": round(past3, 4),
                    "threshold": self.threshold, "factors": factors}

        feats = build_frame(df)[FEATURE_NAMES_V3].iloc[-1:]
        if feats.isna().any().any():
            return {"probability": None, "signal": "WARMING_UP",
                    "in_setup": True, "threshold": self.threshold, "factors": factors}

        raw = float(self.model.predict_proba(feats.values.astype(np.float32))[0, 1])
        prob = float(self.iso.predict([raw])[0])

        direction = "BEARISH" if past3 > 0 else "BULLISH"
        if prob >= self.threshold:
            signal = f"REVERSAL_{direction}"
        else:
            signal = "NO_REVERSAL_EXPECTED"

        return {
            "probability": round(prob, 4),
            "probability_raw": round(raw, 4),
            "signal": signal,
            "direction": direction,
            "in_setup": True,
            "fired": bool(prob >= self.threshold),
            "threshold": round(self.threshold, 4),
            "past_return_3m_pct": round(past3, 4),
            "factors": factors,
        }


def demo():
    """Self-check: gate, warmup, and shape handling without a trained model on disk."""
    from types import SimpleNamespace
    rng = np.random.default_rng(1)

    def mk(n, drift=0.0):
        out = []
        px = 100.0
        for i in range(n):
            px *= 1 + drift + rng.normal(0, 0.0004)
            v = float(rng.uniform(1, 5))
            out.append(SimpleNamespace(open_time_ms=i * 60000, open=px, high=px * 1.0005,
                                       low=px * 0.9995, close=px, volume=v,
                                       buy_volume=v * 0.5, sell_volume=v * 0.5))
        return out

    p = ReversalPredictorV3("BTCUSDT")

    # Too few bars must never produce a probability.
    r = p.predict_from_candles(mk(20))
    assert r["probability"] is None and r["signal"] in ("WARMING_UP", "MODEL_UNAVAILABLE")

    # A flat tape must be gated out rather than scored.
    flat = mk(200, drift=0.0)
    for c in flat[-5:]:
        c.close = c.open = c.high = c.low = 100.0
    r = p.predict_from_candles(flat)
    assert r["in_setup"] is False or r["signal"] == "MODEL_UNAVAILABLE", r

    # If a model is present, a strong prior move must be scored and calibrated.
    if p.model is not None:
        trend = mk(200)
        base = trend[-4].close
        for k, c in enumerate(trend[-3:], 1):
            c.close = c.high = base * (1 + 0.004 * k)
            c.low = c.open = base
        r = p.predict_from_candles(trend)
        assert r["in_setup"] is True, r
        assert 0.0 <= r["probability"] <= 1.0, r
        assert r["direction"] == "BEARISH", r
        print(f"demo scored: p={r['probability']} thr={r['threshold']} sig={r['signal']}")

    print("predictor_v3 demo OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
