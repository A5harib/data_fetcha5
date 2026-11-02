"""
Quick out-of-sample check of Kronos on recent hourly candles: every STEP hours
over the last DAYS days, forecast the next 4 hours and compare with what
happened. Kronos never saw these bars. Small sample, so read it as a smoke test.

    python -m model4.check --days 20 --step 12
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from model4 import forecaster as f
from model4.bot import FEE, THRESHOLD
from model4.data import SYMBOLS, history

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--step", type=int, default=12)
    a = ap.parse_args()
    need = f.CONTEXT + a.days * 24 + f.HORIZON
    data = {s: history(s, need) for s in SYMBOLS}
    n = min(len(v) for v in data.values())
    data = {s: v.iloc[-n:] for s, v in data.items()}
    rows = []
    for end in range(f.CONTEXT, n - f.HORIZON, a.step):
        fc = f.forecast({s: v.iloc[end - f.CONTEXT:end] for s, v in data.items()}, seed=end)
        for s, v in data.items():
            real = 100 * (v["close"].iloc[end + f.HORIZON - 1] / v["close"].iloc[end - 1] - 1)
            rows.append({"symbol": s, "at": str(v.index[end - 1]), "forecast": fc[s], "real": real})
        print(len(rows) // len(SYMBOLS), "steps", flush=True)
    fc = np.array([r["forecast"] for r in rows])
    real = np.array([r["real"] for r in rows])
    calls = np.abs(fc) >= THRESHOLD
    pnl = np.sign(fc[calls]) * real[calls] - 200 * FEE
    out = {
        "points": len(rows), "up_forecasts_pct": float(100 * (fc > 0).mean()), "real_up_pct": float(100 * (real > 0).mean()),
        "direction_right_pct_all": float(100 * (np.sign(fc) == np.sign(real)).mean()),
        "trades": int(calls.sum()), "trade_right_pct": float(100 * (np.sign(fc[calls]) == np.sign(real[calls])).mean()) if calls.any() else None,
        "avg_trade_after_fees_pct": float(pnl.mean()) if calls.any() else None,
    }
    print(json.dumps(out, indent=1))
    json.dump({"summary": out, "rows": rows}, open("data/model4_check.json", "w"))
