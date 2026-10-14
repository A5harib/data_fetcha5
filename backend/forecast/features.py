"""
Features and labels on the canonical 1m frame from forecast.data.

Every feature at row t uses bars <= t only. Bar t is closed when we predict, so
its own close/volume are fair game. test_forecast.py enforces this by
recomputing features on a truncated frame and requiring identical values.

Everything is scale-free (log ratios, z-scores, fractions), so 2022 and 2026
prices, and BTC and gold, look alike to the model.

Labels live at the bottom. They are the only code here that reads forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Longest lookback any feature needs, in bars. Rows before this are dropped.
WARMUP = 1440 * 2

# Reference volatility: EWMA of squared 1m log returns. Returns and labels are
# divided by it, so "a 2-sigma hour" means the same thing in any regime.
VOL_HALFLIFE = 240

FEATURES = [
    # trend, in sigma units at each horizon
    "ret_1", "ret_5", "ret_15", "ret_60", "ret_240", "ret_1440",
    # volatility level (log per-minute vol) and term structure
    "lvol_ref", "lrv_15", "lrv_60", "lrv_240", "lrv_1440",
    "vratio_15_240", "vratio_60_1440", "lrange_60",
    # order flow: taker buy share minus sell share, per window
    "flow_5", "flow_15", "flow_60", "flow_240",
    # activity relative to the trailing day
    "lvolume_15", "lvolume_60", "lcount_15", "lsize_60",
    # where price sits
    "rangepos_60", "rangepos_240", "rangepos_1440", "vwapdist_240", "vwapdist_1440",
    # perp basis and funding
    "premium", "premium_dev_1440", "premium_chg_60", "funding",
    # calendar: intraday vol seasonality, funding settlement cycle, weekends
    "hour_sin", "hour_cos", "weekend", "min_to_funding",
]


def _lr(c: pd.Series) -> pd.Series:
    return np.log(c).diff()


def reference_vol(df: pd.DataFrame) -> pd.Series:
    """Per-minute vol estimate known at the close of each bar."""
    r2 = _lr(df["close"]).pow(2)
    return np.sqrt(r2.ewm(halflife=VOL_HALFLIFE, min_periods=VOL_HALFLIFE).mean()).clip(lower=1e-6)


# Funding settlement interval in hours. Binance moved XAUUSDT to 4h; the rest
# of what is traded here settles every 8h on the 00/08/16 UTC grid.
FUNDING_HOURS = {"XAUUSDT": 4}


def build_features(df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    v, cnt = df["volume"], df["count"]
    lr = _lr(c)
    sig = reference_vol(df)
    out = pd.DataFrame(index=df.index)

    for k in (1, 5, 15, 60, 240, 1440):
        out[f"ret_{k}"] = np.log(c / c.shift(k)) / (sig * np.sqrt(k))

    lsig = np.log(sig)
    out["lvol_ref"] = lsig
    rv = {}
    for k in (15, 60, 240, 1440):
        rv[k] = np.log(np.sqrt(lr.pow(2).rolling(k).mean()).clip(lower=1e-7))
        out[f"lrv_{k}"] = rv[k] - lsig  # relative to reference, level is in lvol_ref
    out["vratio_15_240"] = rv[15] - rv[240]
    out["vratio_60_1440"] = rv[60] - rv[1440]
    rng = np.log(h.rolling(60).max() / l.rolling(60).min())
    out["lrange_60"] = np.log(rng.clip(lower=1e-7)) - lsig - 0.5 * np.log(60)

    buy = df["taker_buy_volume"]
    for k in (5, 15, 60, 240):
        vs = v.rolling(k).sum()
        out[f"flow_{k}"] = ((2 * buy.rolling(k).sum() - vs) / vs).where(vs > 0, 0.0)

    v_day = v.rolling(1440).mean().clip(lower=1e-9)
    n_day = cnt.rolling(1440).mean().clip(lower=1e-9)
    out["lvolume_15"] = np.log((v.rolling(15).mean() / v_day).clip(lower=1e-3))
    out["lvolume_60"] = np.log((v.rolling(60).mean() / v_day).clip(lower=1e-3))
    out["lcount_15"] = np.log((cnt.rolling(15).mean() / n_day).clip(lower=1e-3))
    size_60 = v.rolling(60).sum() / cnt.rolling(60).sum().clip(lower=1)
    size_day = v.rolling(1440).sum() / cnt.rolling(1440).sum().clip(lower=1)
    out["lsize_60"] = np.log((size_60 / size_day.clip(lower=1e-12)).clip(lower=1e-3))

    for k in (60, 240, 1440):
        lo, hi = l.rolling(k).min(), h.rolling(k).max()
        out[f"rangepos_{k}"] = ((c - lo) / (hi - lo)).where(hi > lo, 0.5)
    qv = df["quote_volume"]
    for k in (240, 1440):
        vwap = qv.rolling(k).sum() / v.rolling(k).sum().clip(lower=1e-9)
        out[f"vwapdist_{k}"] = np.log(c / vwap) / (sig * np.sqrt(k))

    p = df["premium"]
    out["premium"] = p * 1e4  # basis points
    out["premium_dev_1440"] = (p - p.rolling(1440).mean()) * 1e4
    out["premium_chg_60"] = (p - p.shift(60)) * 1e4
    out["funding"] = df["funding"] * 1e4

    # Features describe the bar's close, so calendar uses close time.
    t = df.index + pd.Timedelta(minutes=1)
    hr = t.hour + t.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    out["weekend"] = (t.dayofweek >= 5).astype(float)
    fh = FUNDING_HOURS.get(symbol, 8)
    out["min_to_funding"] = (fh * 60 - (t.hour % fh) * 60 - t.minute) % (fh * 60)

    return out[FEATURES].astype("float32")


# ---------------------------------------------------------------- labels

def make_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward targets for a prediction made at the close of bar t:
      ret       log(close[t+H] / close[t])
      ret_z     ret / (sigma_ref[t] * sqrt(H))        direction model target
      lvol_rel  log(realized vol over t+1..t+H) - log(sigma_ref[t])   vol model target
      valid     False if the window is incomplete or crosses filled bars
    sigma_ref[t] is known at t, so dividing by it leaks nothing.
    """
    c = df["close"]
    sig = reference_vol(df)
    lr = _lr(c)
    ret = np.log(c.shift(-horizon) / c)
    # sum of r^2 over (t, t+H]: rolling sum ending at t+H, shifted back to t
    fwd_rv = np.sqrt(lr.pow(2).rolling(horizon).sum().shift(-horizon) / horizon)
    filled_ahead = df["filled"].astype(float).rolling(horizon).sum().shift(-horizon)
    out = pd.DataFrame(index=df.index)
    out["ret"] = ret
    out["ret_z"] = ret / (sig * np.sqrt(horizon))
    out["lvol_rel"] = np.log(fwd_rv.clip(lower=1e-7)) - np.log(sig)
    out["sigma_ref"] = sig
    out["valid"] = ret.notna() & fwd_rv.notna() & (filled_ahead == 0) & (fwd_rv > 0)
    return out


def dataset(df: pd.DataFrame, horizon: int, symbol: str = "BTCUSDT"):
    """Aligned (X, labels) with warmup and invalid rows removed."""
    X = build_features(df, symbol)
    y = make_labels(df, horizon)
    keep = y["valid"] & X.notna().all(axis=1)
    keep.iloc[:WARMUP] = False
    return X[keep], y[keep]
