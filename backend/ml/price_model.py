"""
Forward-return regressor with online retraining.

Why a regressor and not the v3 classifier: the classifier answers "will this
reverse, yes/no" and has no notion of a price. To compare a predicted price
against the price that actually happened, the model has to output a magnitude.
This predicts the forward return over HORIZON bars, which converts to a price
by  predicted_price = last_close * (1 + predicted_return).

Two things worth being blunt about:

1. It predicts the return, never the price directly. Training on price levels
   would let the model "predict" 78,000 by echoing the last close, scoring a
   spectacular R^2 while knowing nothing. Returns force it to earn the number.
   For the same reason the honest baseline to beat is "next price = current
   price" (a random walk), which is already very hard to beat on 1m crypto.

2. Every bar is scored, unlike the v3 classifier which only fires on setups.
   That makes the comparison against actual prices continuous.

Online retraining: `partial_fit`-style refit on a rolling window of the most
recent bars. XGBoost has no true incremental update that keeps a fixed model
size, so this refits on a bounded recent window (WINDOW_BARS) every
RETRAIN_EVERY closed bars. On a few thousand rows and 12 features that is
well under a second, so it comfortably fits between 1-minute bars.
"""
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.features_v3 import FEATURE_NAMES_V3, build_frame, HORIZON

logger = logging.getLogger("price_model")

# Rolling training window. Old regimes actively mislead on 1m crypto, so the
# model deliberately forgets: only the most recent WINDOW_BARS bars train it.
WINDOW_BARS = 5000
# Refit cadence, in closed bars.
RETRAIN_EVERY = 30
# Minimum bars before the model will emit anything.
MIN_TRAIN_BARS = 600

PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=3.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
)


def forward_return_pct(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Return over the next `horizon` bars, in percent. NaN on the unlabeled tail."""
    return (close.shift(-horizon) / close - 1.0) * 100.0


class OnlinePriceModel:
    """
    Rolling-window forward-return regressor.

    Feed it closed bars with `add_bar`; it refits itself every RETRAIN_EVERY
    bars on the most recent WINDOW_BARS. Call `predict_next` for the predicted
    price `horizon` bars ahead.
    """

    def __init__(self, symbol: str, horizon: int = HORIZON,
                 window_bars: int = WINDOW_BARS, retrain_every: int = RETRAIN_EVERY):
        self.symbol = symbol.upper()
        self.horizon = horizon
        self.window_bars = window_bars
        self.retrain_every = retrain_every

        self.bars: deque = deque(maxlen=window_bars)
        self.model: Optional[xgb.XGBRegressor] = None
        self.bars_since_fit = 0
        self.n_fits = 0
        self.last_train_rows = 0
        # Residual spread from the most recent fit, used for the prediction band.
        self.resid_std: float = 0.0

    def add_bar(self, bar: dict) -> bool:
        """
        Append one CLOSED bar. Returns True if this triggered a refit.

        bar keys: open_time, open, high, low, close, volume, buy_volume, sell_volume
        """
        self.bars.append(bar)
        self.bars_since_fit += 1
        if len(self.bars) >= MIN_TRAIN_BARS and self.bars_since_fit >= self.retrain_every:
            self.fit()
            return True
        return False

    def _frame(self) -> pd.DataFrame:
        return build_frame(pd.DataFrame(list(self.bars)))

    def fit(self) -> bool:
        """Refit on the current rolling window. Returns False if not enough clean rows."""
        if len(self.bars) < MIN_TRAIN_BARS:
            return False

        d = self._frame()
        d["_y"] = forward_return_pct(d["close"], self.horizon)
        cols = FEATURE_NAMES_V3 + ["_y"]
        # The last `horizon` bars have no label yet -- dropna removes them, so
        # the model never trains on a future it cannot see.
        d = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < MIN_TRAIN_BARS // 2:
            return False

        X = d[FEATURE_NAMES_V3].values
        y = d["_y"].values

        # Recency weighting: the newest bar counts ~3x the oldest in the window.
        w = np.linspace(1.0, 3.0, len(X))

        m = xgb.XGBRegressor(**PARAMS)
        m.fit(X, y, sample_weight=w)

        self.model = m
        self.n_fits += 1
        self.last_train_rows = len(X)
        self.bars_since_fit = 0
        self.resid_std = float(np.std(y - m.predict(X)))
        return True

    def predict_next(self) -> Optional[dict]:
        """
        Predict the price `horizon` bars ahead of the newest bar held.

        Returns None while warming up. The returned `predicted_price` is what
        the model expects at bar t+horizon, to be compared against the actual
        close when that bar arrives.
        """
        if self.model is None or len(self.bars) < MIN_TRAIN_BARS:
            return None

        d = self._frame()
        row = d[FEATURE_NAMES_V3].iloc[-1:]
        if row.isna().any().any():
            return None

        pred_ret = float(self.model.predict(row.values.astype(np.float32))[0])
        last_close = float(d["close"].iloc[-1])
        return {
            "symbol": self.symbol,
            "from_open_time": int(d["open_time"].iloc[-1]),
            "last_close": last_close,
            "predicted_return_pct": pred_ret,
            "predicted_price": last_close * (1.0 + pred_ret / 100.0),
            "horizon_bars": self.horizon,
            "band_pct": self.resid_std,
            "n_fits": self.n_fits,
            "train_rows": self.last_train_rows,
        }


def demo():
    """Self-check: no lookahead in labels, refit cadence, and price reconstruction."""
    rng = np.random.default_rng(7)
    n = 1200
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    bars = [{"open_time": i * 60000, "open": float(close[i]), "high": float(close[i]) + .05,
             "low": float(close[i]) - .05, "close": float(close[i]),
             "volume": float(rng.uniform(1, 9)), "buy_volume": 0.0, "sell_volume": 0.0}
            for i in range(n)]
    for b in bars:
        b["buy_volume"] = b["volume"] * 0.55
        b["sell_volume"] = b["volume"] - b["buy_volume"]

    # Label must look strictly forward and leave the tail unlabeled.
    s = pd.Series([b["close"] for b in bars])
    fr = forward_return_pct(s, 3)
    assert abs(fr.iloc[10] - (s.iloc[13] / s.iloc[10] - 1) * 100) < 1e-9, "label misaligned"
    assert fr.iloc[-3:].isna().all(), "tail must stay unlabeled"

    m = OnlinePriceModel("TEST", horizon=3, window_bars=800, retrain_every=50)
    fits = sum(m.add_bar(b) for b in bars)
    assert fits >= 1 and m.n_fits == fits, (fits, m.n_fits)
    assert len(m.bars) == 800, "rolling window must be bounded"

    p = m.predict_next()
    assert p is not None, "should predict once warm"
    # predicted_price must be the reconstruction of predicted_return_pct.
    exp = p["last_close"] * (1 + p["predicted_return_pct"] / 100.0)
    assert abs(p["predicted_price"] - exp) < 1e-9, "price reconstruction wrong"
    # On random-walk input the model must not claim a large edge.
    assert abs(p["predicted_return_pct"]) < 5.0, p

    print(f"price_model demo OK (fits={m.n_fits}, window={len(m.bars)}, "
          f"pred={p['predicted_price']:.2f} vs last={p['last_close']:.2f})")


if __name__ == "__main__":
    demo()
