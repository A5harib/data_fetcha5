"""
Cost-aware backtest of the reversal signal.

The classifier's 35% hit rate says nothing about money. This asks the only
question that matters: after fees, slippage and a stop, does trading the
signal beat not trading it?

Assumptions, all conservative on purpose:

- Entry at the NEXT bar's open, never the signal bar's close. You cannot fill
  at a price that has already printed.
- Taker fees both sides. Binance USDT-M futures taker is 0.04% per side, so
  0.08% round trip, against a target of only 0.20%. Fees are 40% of the gross
  target -- this is the whole reason the backtest exists.
- Slippage of one tick each way, plus a configurable spread cost.
- WITHIN-BAR AMBIGUITY IS RESOLVED AGAINST THE TRADE. When a bar's high and
  low would have touched both the target and the stop, 1-minute OHLC cannot
  say which came first, so the backtest records the stop. Assuming the target
  filled first is the single most common way a backtest lies.
- A trade that reaches neither level exits at the close of the last bar in the
  horizon window.

Baselines it must beat:
  hold      buy and hold over the same span
  random    same number of trades, same holding time, entered at random bars

Usage:
    python ml/backtest.py BTCUSDT
    python ml/backtest.py BTCUSDT --stop 0.15 --horizon 5 --fee 0.04
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from config import settings
from ml.features_v3 import build_dataset, HORIZON
from ml.train_v3 import load_local_klines, _fit_one, N_FOLDS, EMBARGO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

TAKER_FEE_PCT = 0.04      # per side, Binance USDT-M futures taker
SLIPPAGE_TICKS = 1
TICK = {"BTCUSDT": 0.10, "XAUUSDT": 0.01}


@dataclass
class Trade:
    entry_time: int
    direction: str
    entry: float
    exit: float
    gross_pct: float
    net_pct: float
    outcome: str      # target | stop | timeout


def simulate(df: pd.DataFrame, signals: pd.DataFrame, symbol: str,
             target_pct: float, stop_pct: float, horizon: int,
             fee_pct: float = TAKER_FEE_PCT) -> list[Trade]:
    """
    signals: DataFrame with columns [open_time, direction].
    Each signal enters at the next bar's open and exits on target, stop, or timeout.
    """
    bars = df.reset_index(drop=True)
    idx = {t: i for i, t in enumerate(bars["open_time"].values)}
    tick = TICK.get(symbol.upper(), 0.01)
    trades: list[Trade] = []

    for _, s in signals.iterrows():
        i = idx.get(int(s["open_time"]))
        # Need the entry bar plus at least one bar to resolve in.
        if i is None or i + 2 > len(bars):
            continue

        entry_bar = bars.iloc[i + 1]
        # Enter at next open, pay slippage against ourselves.
        long = s["direction"] == "BULLISH"
        entry = float(entry_bar["open"]) + (tick * SLIPPAGE_TICKS if long else -tick * SLIPPAGE_TICKS)

        tgt = entry * (1 + target_pct / 100) if long else entry * (1 - target_pct / 100)
        stp = entry * (1 - stop_pct / 100) if long else entry * (1 + stop_pct / 100)

        exit_px, outcome = None, "timeout"
        for j in range(i + 1, min(i + 1 + horizon, len(bars))):
            b = bars.iloc[j]
            hi, lo = float(b["high"]), float(b["low"])
            hit_t = (hi >= tgt) if long else (lo <= tgt)
            hit_s = (lo <= stp) if long else (hi >= stp)
            if hit_t and hit_s:
                # Ambiguous inside the bar. Assume the stop filled first.
                exit_px, outcome = stp, "stop"
                break
            if hit_s:
                exit_px, outcome = stp, "stop"
                break
            if hit_t:
                exit_px, outcome = tgt, "target"
                break
        if exit_px is None:
            exit_px = float(bars.iloc[min(i + horizon, len(bars) - 1)]["close"])
            outcome = "timeout"

        # Slippage on exit too.
        exit_px = exit_px - tick * SLIPPAGE_TICKS if long else exit_px + tick * SLIPPAGE_TICKS

        gross = (exit_px / entry - 1) * 100 * (1 if long else -1)
        net = gross - 2 * fee_pct
        trades.append(Trade(int(s["open_time"]), s["direction"], entry, exit_px,
                            gross, net, outcome))
    return trades


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    net = np.array([t.net_pct for t in trades])
    gross = np.array([t.gross_pct for t in trades])
    wins = net > 0
    return {
        "n": len(trades),
        "win_rate": float(wins.mean()),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "net_total": float(net.sum()),
        "net_median": float(np.median(net)),
        "best": float(net.max()), "worst": float(net.min()),
        "target": sum(t.outcome == "target" for t in trades),
        "stop": sum(t.outcome == "stop" for t in trades),
        "timeout": sum(t.outcome == "timeout" for t in trades),
        # Per-trade Sharpe-like ratio; not annualized, just mean/std.
        "sharpe": float(net.mean() / net.std()) if net.std() > 0 else float("nan"),
    }


def walk_forward_signals(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals the honest way: every signal comes from a model that never
    saw that bar. Reuses the same purged walk-forward split as training.
    """
    thr_pct = settings.reversal_threshold_for(symbol)
    gate = settings.gate_for(symbol)
    X, y, ot = build_dataset(df, thr_pct, gated=True, gate_pct=gate)
    X, y, ot = X.reset_index(drop=True), y.reset_index(drop=True), ot.reset_index(drop=True)

    # Direction is implied by the prior move, same rule the predictor uses.
    close = df.set_index("open_time")["close"]
    past3 = {}
    c = df["close"].values
    times = df["open_time"].values
    for i in range(3, len(df)):
        past3[int(times[i])] = (c[i] / c[i - 3] - 1) * 100

    n = len(X)
    start = int(n * 0.40)
    fold = (n - start) // N_FOLDS
    rows = []
    for k in range(N_FOLDS):
        lo = start + k * fold
        hi = lo + fold if k < N_FOLDS - 1 else n
        tr_hi = lo - EMBARGO
        if tr_hi < 500 or hi - lo < 50:
            continue
        ytr, yte = y.iloc[:tr_hi].values, y.iloc[lo:hi].values
        if ytr.sum() < 30 or yte.sum() < 5:
            continue
        _, _, _, p, thr = _fit_one(X.iloc[:tr_hi].values, ytr,
                                   X.iloc[lo:hi].values, yte)
        for m, prob in enumerate(p):
            if prob >= thr:
                t = int(ot.iloc[lo + m])
                pm = past3.get(t)
                if pm is None:
                    continue
                rows.append({"open_time": t, "prob": float(prob),
                             "direction": "BEARISH" if pm > 0 else "BULLISH"})
    return pd.DataFrame(rows)


def random_baseline(df: pd.DataFrame, symbol: str, n_trades: int, target_pct: float,
                    stop_pct: float, horizon: int, fee: float, seed: int = 0) -> dict:
    """Same trade count and mechanics, entered at random bars with random direction."""
    rng = np.random.default_rng(seed)
    valid = df["open_time"].values[100:-(horizon + 2)]
    outs = []
    for s in range(20):                      # average over 20 random draws
        pick = rng.choice(valid, size=min(n_trades, len(valid)), replace=False)
        sig = pd.DataFrame({"open_time": pick,
                            "direction": rng.choice(["BULLISH", "BEARISH"], len(pick))})
        outs.append(stats(simulate(df, sig, symbol, target_pct, stop_pct, horizon, fee)))
    return {k: float(np.mean([o[k] for o in outs if o.get("n")]))
            for k in ("win_rate", "net_mean", "net_total")}


def demo():
    """Self-check: fee arithmetic, stop-before-target rule, direction handling."""
    # A clean +1% long with zero fee and no stop must return ~+1% gross.
    bars = pd.DataFrame({
        "open_time": [0, 60000, 120000, 180000],
        "open": [100.0, 100.0, 100.0, 100.0],
        "high": [100.0, 102.0, 102.0, 102.0],
        "low": [100.0, 100.0, 100.0, 100.0],
        "close": [100.0, 101.0, 101.0, 101.0],
    })
    sig = pd.DataFrame([{"open_time": 0, "direction": "BULLISH"}])
    t = simulate(bars, sig, "TESTUSDT", target_pct=1.0, stop_pct=5.0, horizon=3, fee_pct=0.0)
    assert len(t) == 1 and t[0].outcome == "target", t
    # Target is 1%, minus a tick of slippage on each side.
    assert 0.9 < t[0].gross_pct <= 1.0, t[0].gross_pct

    # Fees must come straight off the net.
    t2 = simulate(bars, sig, "TESTUSDT", target_pct=1.0, stop_pct=5.0, horizon=3, fee_pct=0.04)
    assert abs((t[0].net_pct - t2[0].net_pct) - 0.08) < 1e-9, "fee arithmetic wrong"

    # A bar touching BOTH target and stop must record the stop, not the target.
    amb = pd.DataFrame({
        "open_time": [0, 60000],
        "open": [100.0, 100.0], "high": [100.0, 102.0],
        "low": [100.0, 98.0], "close": [100.0, 100.0],
    })
    t3 = simulate(amb, pd.DataFrame([{"open_time": 0, "direction": "BULLISH"}]),
                  "TESTUSDT", target_pct=1.0, stop_pct=1.0, horizon=1, fee_pct=0.0)
    assert t3[0].outcome == "stop", "ambiguous bar must resolve against the trade"
    assert t3[0].gross_pct < 0, t3[0].gross_pct

    # A short must profit when price falls.
    down = pd.DataFrame({
        "open_time": [0, 60000], "open": [100.0, 100.0],
        "high": [100.0, 100.0], "low": [100.0, 99.0], "close": [100.0, 99.0],
    })
    t4 = simulate(down, pd.DataFrame([{"open_time": 0, "direction": "BEARISH"}]),
                  "TESTUSDT", target_pct=1.0, stop_pct=5.0, horizon=1, fee_pct=0.0)
    assert t4[0].gross_pct > 0, t4[0].gross_pct
    print("backtest demo OK")


def main(symbol: str, target: float, stop: float, horizon: int, fee: float):
    df = load_local_klines(symbol)
    sig = walk_forward_signals(symbol, df)
    if sig.empty:
        print("no signals generated")
        return

    trades = simulate(df, sig, symbol, target, stop, horizon, fee)
    s = stats(trades)
    span_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    rnd = random_baseline(df, symbol, s["n"], target, stop, horizon, fee)

    print("\n" + "=" * 74)
    print(f"{symbol}  target {target}%  stop {stop}%  horizon {horizon}m  "
          f"fee {fee}%/side")
    print(f"{len(df)} bars, buy-and-hold over the same span: {span_pct:+.2f}%")
    print("=" * 74)
    print(f"  trades            {s['n']}")
    print(f"  win rate          {s['win_rate']:.1%}")
    print(f"  target/stop/timeout   {s['target']}/{s['stop']}/{s['timeout']}")
    print(f"  gross per trade   {s['gross_mean']:+.4f}%")
    print(f"  net per trade     {s['net_mean']:+.4f}%   <- after {2*fee:.2f}% round-trip cost")
    print(f"  net total         {s['net_total']:+.2f}%")
    print(f"  best / worst      {s['best']:+.2f}% / {s['worst']:+.2f}%")
    print(f"  per-trade sharpe  {s['sharpe']:+.3f}")
    print(f"\n  random baseline   win {rnd['win_rate']:.1%}  "
          f"net/trade {rnd['net_mean']:+.4f}%  total {rnd['net_total']:+.2f}%")
    edge = s["net_mean"] - rnd["net_mean"]
    print(f"  edge over random  {edge:+.4f}% per trade")
    if s["net_mean"] <= 0:
        print("\n  LOSES MONEY after costs. The hit rate is real but the target is")
        print("  too small relative to fees and the stop.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--target", type=float, default=None, help="take-profit %%")
    ap.add_argument("--stop", type=float, default=None, help="stop-loss %%")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--fee", type=float, default=TAKER_FEE_PCT)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        sym = a.symbol.upper()
        tgt = a.target if a.target is not None else settings.reversal_threshold_for(sym)
        stp = a.stop if a.stop is not None else tgt
        main(sym, tgt, stp, a.horizon, a.fee)
