"""
Replay spot history through the model 2 agent and save what it learned.

    python -m model2.data  --symbol BTCUSDT      # download spot history once
    python -m model2.train --symbol BTCUSDT

The replay is the backtest: at every minute the agent acts on weights learned
only from the past, the same loop it runs live. The scaler is the one thing
fit up front, on the first 120 days after warmup, and the backtest starts after
that. The paper account is reset before saving, so the live run starts at $100.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model2.agent import FEE, WARMUP, Agent, advance, features
from model2.data import load

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models")) / "model2"


def report(trades: list) -> dict:
    if not trades:
        print("no trades")
        return {}
    t = pd.DataFrame(trades)
    t["year"] = pd.to_datetime(t["entry_t"] * 60, unit="s").dt.year
    t["net"] = (1 - FEE) ** 2 * (1 + t["move_pct"] / 100) - 1
    out = {}
    for yr, g in t.groupby("year"):
        grow = float(np.prod(1 + g["net"]))
        out[int(yr)] = {"trades": len(g), "right": round(float(g["right"].mean()), 3),
                        "hours_held": round(float((g["exit_t"] - g["entry_t"]).mean() / 60), 2),
                        "per_100": round(100 * grow, 2),
                        "per_100_no_fees": round(100 * float(np.prod(1 + g["move_pct"] / 100)), 2)}
        o = out[int(yr)]
        print(f"  {yr}: {o['trades']:4d} trades, right {o['right']:.0%}, held {o['hours_held']:.1f}h avg, "
              f"$100 -> ${o['per_100']:.2f} after fees (${o['per_100_no_fees']:.2f} without)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    a = ap.parse_args()
    sym = a.symbol.upper()
    t0 = time.time()
    df = load(sym)
    X = features(df, sym)
    A = X.to_numpy(float)
    ok = np.flatnonzero(np.isfinite(A).all(axis=1))
    ok = ok[ok >= WARMUP]
    fit_rows = ok[:120 * 1440]
    agent = Agent(sym, Agent.fit_scaler(A[fit_rows]))
    start = int(fit_rows[-1]) + 1
    print(f"{sym}: {len(df):,} bars, scaler from {df.index[fit_rows[0]].date()} .. {df.index[fit_rows[-1]].date()}, "
          f"backtest from {df.index[start].date()} ({time.time() - t0:.0f}s)")

    # Replay the scaler months too, so the agent starts the backtest already
    # warmed up, but only count paper trades from `start` on.
    pre = df.index[fit_rows[0]:start]
    advance(agent, df.loc[pre], X.loc[pre])
    agent.start_real(100.0, str(df.index[start]))
    advance(agent, df, X)
    print(f"replayed in {time.time() - t0:.0f}s, learned from {agent.hours_learned:,} hours")
    print("Backtest, paper account following the self-tuned agent:")
    yearly = report(agent.real["trades"])

    trained_through = str(df.index[-1])
    agent.real = None  # live run starts fresh
    for L in agent.levels.values():
        L.pop("live", None)  # backtest calls must not count toward the live record
    out = MODEL_DIR / f"{sym}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(agent.dumps())
    os.replace(tmp, out)
    meta = {"symbol": sym, "trained_through": trained_through, "backtest": yearly,
            "hours_learned": agent.hours_learned}
    (MODEL_DIR / f"{sym}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
