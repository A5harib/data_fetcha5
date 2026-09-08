"""
Feature engineering and labeling, v3.

Fixes three defects in ml/feature_pipeline.py that made the reported AUC of
0.96 meaningless:

1. The old label ANDed a *past* condition (|past_return_3m| > 0.2) with a
   future one. 96.4% of rows failed the past condition and could never be
   positive, so the gate alone scored AUC 0.9878 -- higher than the trained
   model. The model was scored almost entirely on recalling the recent past.
   Here the gate becomes a row filter, applied before training, so every row
   in the dataset is one where a reversal is genuinely possible and the base
   rate is ~0.26 instead of ~0.009.

2. `orderbook_imbalance` was synthesized from kline buy/sell volume whenever
   the column was absent, which is always the training path. That made it an
   exact rescale of cvd_1m_delta, and it did not match serving, where the
   value comes from live L2 depth. It is gone from the training feature set.

3. The six old features carried ~0.03 AUC each. v3 adds realized volatility,
   volume z-score, CVD/price divergence, ATR-relative bar range, longer-horizon
   returns, and time-of-day.
"""
import sys
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

# Feature order is the contract between training and serving. Changing it
# invalidates every saved model.
FEATURE_NAMES_V3 = [
    # Volatility state. realized_vol_15m is the single strongest feature on both
    # symbols by leave-one-out ablation (+0.013 AUC BTC, +0.027 XAU).
    "realized_vol_15m",
    "vol_ratio_60m",
    "range_atr_ratio",
    "volume_z",
    # Bar shape: where price closed inside its range. Rejection wicks are the
    # classic reversal tell and ablate positively on BTC.
    "body_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    # Order flow against price.
    "cvd_price_divergence",
    "cvd_5m_z",
    # Location and momentum.
    "vwap_distance_pct",
    "rsi_14",
    "minute_cos",
]

# Dropped after leave-one-out ablation on purged walk-forward folds: cvd_1m_z,
# price_change_pct, ret_5m_pct, ret_15m_pct, vwap_dist_z, minute_sin. Each
# scored a NEGATIVE contribution on both symbols -- removing them raised AUC.
# On ~1.6k BTC rows the model cannot afford features that only add variance.
DROPPED_V3 = [
    "cvd_1m_z", "price_change_pct", "ret_5m_pct",
    "ret_15m_pct", "vwap_dist_z", "minute_sin",
]

# Gate: only bars following a meaningful 3-bar move can host a reversal.
GATE_PAST_RETURN_PCT = 0.2
# Forward horizon in bars.
HORIZON = 3


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / (sd + 1e-9)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all v3 features. Every column uses only data at or before its bar."""
    d = df.copy()
    close, high, low = d["close"], d["high"], d["low"]
    vol = d["volume"]

    delta = d["buy_volume"] - d["sell_volume"]
    d["cvd_1m_z"] = _zscore(delta, 60)
    d["cvd_5m_z"] = _zscore(delta.rolling(5, min_periods=1).sum(), 60)

    # Divergence: normalized CVD push against normalized price push. Large
    # negative means aggressive flow is not moving price -- absorption.
    cvd_5 = delta.rolling(5, min_periods=1).sum()
    ret_5 = close.pct_change(5) * 100.0
    d["cvd_price_divergence"] = _zscore(cvd_5, 120) - _zscore(ret_5, 120)

    d["price_change_pct"] = close.pct_change() * 100.0
    d["ret_5m_pct"] = ret_5
    d["ret_15m_pct"] = close.pct_change(15) * 100.0

    d["rsi_14"] = _rsi(close, 14)

    tp = (high + low + close) / 3.0
    cvp = (tp * vol).rolling(60, min_periods=10).sum()
    cv = vol.rolling(60, min_periods=10).sum()
    vwap = np.where(cv > 0, cvp / cv, close)
    d["vwap_distance_pct"] = (close - vwap) / vwap * 100.0
    d["vwap_dist_z"] = _zscore(d["vwap_distance_pct"], 120)

    r1 = close.pct_change() * 100.0
    d["realized_vol_15m"] = r1.rolling(15, min_periods=5).std()
    d["vol_ratio_60m"] = d["realized_vol_15m"] / (r1.rolling(60, min_periods=20).std() + 1e-9)
    d["volume_z"] = _zscore(vol, 60)

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    d["range_atr_ratio"] = (high - low) / (atr + 1e-9)

    rng = (high - low).replace(0, np.nan)
    d["body_frac"] = (close - d["open"]).abs() / rng
    d["upper_wick_frac"] = (high - close.combine(d["open"], max)) / rng
    d["lower_wick_frac"] = (close.combine(d["open"], min) - low) / rng

    # Time of day. Gold and crypto both have session structure.
    ts = pd.to_datetime(d["open_time"], unit="ms")
    minute = ts.dt.hour * 60 + ts.dt.minute
    d["minute_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    d["minute_cos"] = np.cos(2 * np.pi * minute / 1440.0)

    return d


def build_dataset(df: pd.DataFrame, reversal_threshold_pct: float,
                  gated: bool = True,
                  gate_pct: float = GATE_PAST_RETURN_PCT
                  ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (X, y, open_time).

    y = 1 when price reverses against the prior 3-bar move by at least
    `reversal_threshold_pct` within the next HORIZON bars.

    With gated=True (the default and the honest setting) only bars that follow
    a >= GATE_PAST_RETURN_PCT 3-bar move are returned, so the model is scored
    on the reversal question alone rather than on detecting the prior move.
    """
    d = build_frame(df)

    past3 = d["close"].pct_change(3) * 100.0
    fwd_max = d["high"].shift(-1).rolling(HORIZON, min_periods=1).max().shift(-(HORIZON - 1))
    fwd_min = d["low"].shift(-1).rolling(HORIZON, min_periods=1).min().shift(-(HORIZON - 1))

    drop_pct = (fwd_min - d["close"]) / d["close"] * 100.0
    rise_pct = (fwd_max - d["close"]) / d["close"] * 100.0

    up_before = past3 > gate_pct
    down_before = past3 < -gate_pct

    d["target"] = ((up_before & (drop_pct <= -reversal_threshold_pct)) |
                   (down_before & (rise_pct >= reversal_threshold_pct))).astype(int)
    d["_gate"] = up_before | down_before

    cols = FEATURE_NAMES_V3 + ["target", "_gate", "open_time"]
    d = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if gated:
        d = d[d["_gate"]]
    d = d.reset_index(drop=True)

    return d[FEATURE_NAMES_V3], d["target"], d["open_time"]


def extract_features_v3(d: dict) -> np.ndarray:
    """Serving-side extraction. Keys absent from the live dict fall back to a neutral value."""
    defaults = {"rsi_14": 50.0}
    return np.array([float(d.get(k, defaults.get(k, 0.0))) for k in FEATURE_NAMES_V3],
                    dtype=np.float32)


def demo():
    """Self-check: label alignment, no lookahead in features, gate behavior."""
    rng = np.random.default_rng(0)
    n = 3000
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame({
        "open_time": np.arange(n) * 60000,
        "open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
        "volume": rng.uniform(1, 10, n),
    })
    df["buy_volume"] = df["volume"] * rng.uniform(0.3, 0.7, n)
    df["sell_volume"] = df["volume"] - df["buy_volume"]

    # Forward window must look strictly ahead: bar i uses highs i+1..i+3.
    f = df["high"].shift(-1).rolling(3, min_periods=1).max().shift(-2)
    assert abs(f.iloc[10] - df["high"].iloc[11:14].max()) < 1e-9, "forward window misaligned"

    X, y, _ = build_dataset(df, 0.15, gated=True)
    assert list(X.columns) == FEATURE_NAMES_V3, "feature order drifted"
    assert len(X) == len(y) > 0, "empty dataset"
    assert not X.isna().any().any(), "NaNs survived"

    # Gating must not dilute positives.
    Xu, yu, _ = build_dataset(df, 0.15, gated=False)
    assert yu.mean() <= y.mean() + 1e-12, "gating did not concentrate positives"

    # Features must not depend on future bars: truncating the frame must not
    # change earlier rows.
    a = build_frame(df).iloc[:1000]
    b = build_frame(df.iloc[:1000])
    pd.testing.assert_frame_equal(
        a[FEATURE_NAMES_V3].iloc[100:900].reset_index(drop=True),
        b[FEATURE_NAMES_V3].iloc[100:900].reset_index(drop=True),
        check_exact=False, rtol=1e-9)

    assert extract_features_v3({}).shape == (len(FEATURE_NAMES_V3),)
    print("features_v3 demo OK")


if __name__ == "__main__":
    demo()
