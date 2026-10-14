"""
Live forecast: same data path and feature code as training, recomputed once per
closed bar.

    GET /api/forecast/{symbol}?horizon=60

    python -m forecast.predictor BTCUSDT 60     # one forecast, printed
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time

import numpy as np
import xgboost as xgb
from fastapi import APIRouter, HTTPException

from forecast.data import fetch_recent
from forecast.direction import DirectionModel
from forecast.features import build_features, reference_vol
from forecast.train import MODEL_DIR

logger = logging.getLogger("forecast.predictor")


class Forecaster:
    def __init__(self, symbol: str, horizon: int):
        self.symbol, self.horizon = symbol.upper(), horizon
        d = MODEL_DIR / f"{self.symbol}_{horizon}m"
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No forecaster for {self.symbol} {horizon}m. Train: "
                                    f"python -m forecast.train --symbol {self.symbol} --horizon {horizon}")
        self.meta = json.loads((d / "meta.json").read_text())
        self.vol = xgb.XGBRegressor()
        self.vol.load_model(d / "vol.json")
        self.dir = None
        if self.meta["ship"]["dir"] and (d / "dir.json").exists():
            self.dir = DirectionModel.from_json((d / "dir.json").read_text())
        self._cache: dict | None = None
        self._lock = threading.Lock()

    def forecast(self, df=None) -> dict:
        if df is None:
            df = fetch_recent(self.symbol)
        X = build_features(df, self.symbol)[self.meta["features"]].iloc[[-1]]
        if X.isna().any(axis=None):
            raise RuntimeError("not enough history for every feature")
        price = float(df["close"].iloc[-1])
        sig = float(reference_vol(df).iloc[-1])
        h = self.horizon
        sigma_h = float(np.exp(self.vol.predict(X)[0]) * sig * np.sqrt(h))  # log-return sd over the horizon
        half = self.meta["k80"] * sigma_h
        out = {
            "symbol": self.symbol,
            "horizon_min": h,
            "as_of": (df.index[-1] + np.timedelta64(1, "m")).isoformat(),  # close of the last bar
            "price": price,
            "volatility": {
                "sigma_pct": round(100 * sigma_h, 4),
                # 80% of out-of-sample moves landed inside this band
                "range80": [round(price * np.exp(-half), 2), round(price * np.exp(half), 2)],
                "range80_pct": round(100 * half, 4),
                # one-sided: price ended above floor90 / below ceiling90 in 9 of 10 past cases
                "floor90": round(price * np.exp(self.meta["zq"]["0.1"] * sigma_h), 2),
                "ceiling90": round(price * np.exp(self.meta["zq"]["0.9"] * sigma_h), 2),
            },
            "direction": None,
            "model": {"trained_through": self.meta["trained_through"], "oos": self.meta["oos"]},
        }
        if self.dir is not None:
            c = float(self.dir.call(X)[0])
            o = self.meta["oos"]
            out["direction"] = {
                # "UP"/"DOWN" only when price is stretched far enough for the
                # snap-back rule to fire; most of the time this is None.
                "call": {1.0: "UP", -1.0: "DOWN"}.get(c),
                "strength": round(abs(float(self.dir.score(X)[0])) / self.dir.threshold, 3),
                "past_hit_rate": round(o["dir_hit"], 4),
                "past_hit_rate_always_up": round(o["dir_hit_always_up"], 4),
                "past_avg_win_pct": round(o["dir_win_bp"] / 100, 4),
                "past_avg_loss_pct": round(o["dir_loss_bp"] / 100, 4),
            }
        return out

    def cached(self) -> dict:
        """Recompute only when a new bar has closed (plus a few seconds for REST to catch up)."""
        with self._lock:
            now = time.time()
            if self._cache is None or now >= self._cache["_next"]:
                f = self.forecast()
                bar_close = np.datetime64(f["as_of"].replace("+00:00", "")).astype("datetime64[s]").astype(int)
                f["_next"] = max(bar_close + 63, now + 5)
                self._cache = f
            return {k: v for k, v in self._cache.items() if k != "_next"}


router = APIRouter()
_forecasters: dict[tuple[str, int], Forecaster] = {}


def get_forecaster(symbol: str, horizon: int) -> Forecaster:
    key = (symbol.upper(), horizon)
    if key not in _forecasters:
        _forecasters[key] = Forecaster(*key)
    return _forecasters[key]


@router.get("/api/forecast/{symbol}")
def forecast_endpoint(symbol: str, horizon: int = 60):
    # Plain def: FastAPI runs it in a worker thread, so the blocking REST calls
    # do not stall the websocket broadcast loop.
    try:
        return get_forecaster(symbol, horizon).cached()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("forecast failed")
        raise HTTPException(503, f"forecast unavailable: {e}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    hz = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print(json.dumps(get_forecaster(sym, hz).forecast(), indent=2, default=float))
