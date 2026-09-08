"""
Live predicted-vs-actual tracker against the real Binance API.

Each closed 1m bar, the model predicts the price HORIZON bars ahead. When that
bar actually arrives, the prediction is scored against the real close and the
row is printed. The model retrains on a rolling window of the newest bars as
they stream in, so it is always fitted to recent tape rather than to the
static JSON snapshot in data/.

The benchmark that matters is the random walk: "the price in 3 minutes will be
the price now". On 1m crypto that is a genuinely hard baseline. A model that
cannot beat it has no edge, however good its raw error looks -- an MAE of
0.04% sounds tiny but is worthless if naive scores 0.038%.

Usage:
    python ml/live_track.py BTCUSDT              # live, runs until Ctrl-C
    python ml/live_track.py BTCUSDT --replay 800 # offline dry run on stored bars
    python ml/live_track.py XAUUSDT --minutes 60
"""
import argparse
import json
import logging
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from config import settings
from ml.price_model import OnlinePriceModel, MIN_TRAIN_BARS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_track")

REST = settings.BINANCE_FUTURES_REST_URL
WARMUP_BARS = 1500          # history pulled before going live
POLL_SEC = 5


def fetch_klines(symbol: str, limit: int = 500, end_time: int | None = None) -> list[dict]:
    """Closed 1m klines from Binance futures REST, oldest first."""
    url = f"{REST}/fapi/v1/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    if end_time:
        url += f"&endTime={end_time}"
    with urllib.request.urlopen(url, timeout=20) as r:
        raw = json.loads(r.read())
    out = []
    for k in raw:
        vol, buy = float(k[5]), float(k[9])
        out.append({"open_time": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": vol,
                    "buy_volume": buy, "sell_volume": max(0.0, vol - buy),
                    "close_time": int(k[6])})
    return out


def warmup_history(symbol: str, want: int) -> list[dict]:
    """Page backwards until `want` bars are collected."""
    bars, end = [], None
    while len(bars) < want:
        chunk = fetch_klines(symbol, limit=min(1000, want - len(bars) + 1), end_time=end)
        if not chunk:
            break
        bars = chunk + bars
        end = chunk[0]["open_time"] - 1
        if len(chunk) < 2:
            break
    # Drop the final bar: it may still be forming.
    return bars[-want:][:-1]


class Scorer:
    """Holds open predictions until their target bar closes, then scores them."""

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.pending: deque = deque()
        self.rows: list[dict] = []

    def submit(self, pred: dict, target_open_time: int):
        self.pending.append((target_open_time, pred))

    def resolve(self, bar: dict) -> dict | None:
        """If this bar closes an open prediction, score it and return the row."""
        if not self.pending:
            return None
        tgt, pred = self.pending[0]
        if bar["open_time"] < tgt:
            return None
        self.pending.popleft()
        if bar["open_time"] != tgt:
            return None  # gap in the tape; drop rather than mis-score

        actual = bar["close"]
        base = pred["last_close"]
        row = {
            "time": bar["open_time"],
            "from_price": base,
            "predicted": pred["predicted_price"],
            "actual": actual,
            "pred_ret": pred["predicted_return_pct"],
            "actual_ret": (actual / base - 1.0) * 100.0,
            "naive_err": abs(actual - base),
            "model_err": abs(actual - pred["predicted_price"]),
            "beta": pred.get("beta", 0.0),
            "raw_ret": pred.get("raw_return_pct", 0.0),
        }
        # Directional hit only counts when the model committed to a direction.
        row["dir_ok"] = (np.sign(row["pred_ret"]) == np.sign(row["actual_ret"])
                         and abs(row["pred_ret"]) > 1e-9)
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        if not self.rows:
            return {}
        m = np.array([r["model_err"] for r in self.rows])
        nv = np.array([r["naive_err"] for r in self.rows])
        pr = np.array([r["pred_ret"] for r in self.rows])
        ar = np.array([r["actual_ret"] for r in self.rows])
        moved = np.abs(ar) > 1e-9
        return {
            "n": len(self.rows),
            "model_mae": float(m.mean()),
            "naive_mae": float(nv.mean()),
            "skill": float(1 - m.mean() / nv.mean()) if nv.mean() else float("nan"),
            "model_mae_bps": float((m / np.array([r["from_price"] for r in self.rows])).mean() * 1e4),
            "naive_mae_bps": float((nv / np.array([r["from_price"] for r in self.rows])).mean() * 1e4),
            "dir_acc": float(np.mean([r["dir_ok"] for r in self.rows])),
            "corr": float(np.corrcoef(pr, ar)[0, 1]) if moved.sum() > 2 else float("nan"),
            "wins": int((m < nv).sum()),
        }


def print_row(sym: str, r: dict, n: int):
    hit = "OK " if r["dir_ok"] else "-- "
    better = "model" if r["model_err"] < r["naive_err"] else "naive"
    print(f"[{n:>4}] {sym}  pred {r['predicted']:>12,.2f}  actual {r['actual']:>12,.2f}  "
          f"err {r['model_err']:>8.2f}  naive {r['naive_err']:>8.2f}  "
          f"{hit}({r['pred_ret']:+.3f}% vs {r['actual_ret']:+.3f}%)  b={r['beta']:.2f}  {better}")


def print_summary(sym: str, s: dict, model: OnlinePriceModel):
    if not s:
        print("no resolved predictions yet")
        return
    print("\n" + "=" * 78)
    print(f"{sym}  n={s['n']}  refits={model.n_fits}  train_rows={model.last_train_rows}  "
          f"beta={model.beta:.3f}")
    print("=" * 78)
    print(f"  model MAE   {s['model_mae']:>10.2f}   ({s['model_mae_bps']:.2f} bps)")
    print(f"  naive MAE   {s['naive_mae']:>10.2f}   ({s['naive_mae_bps']:.2f} bps)")
    print(f"  skill vs naive   {s['skill']:>+7.2%}   (positive = beats random walk)")
    print(f"  directional acc  {s['dir_acc']:>7.2%}   (50% = coin flip)")
    print(f"  corr(pred, actual returns)  {s['corr']:>+.4f}")
    print(f"  bars where model beat naive: {s['wins']}/{s['n']}")
    if s["skill"] <= 0:
        print("\n  Model does not beat the random walk. On 1m horizons that is the")
        print("  normal outcome -- the naive forecast is very strong here.")


def run(symbol: str, minutes: int | None, replay: int | None):
    horizon = 3
    model = OnlinePriceModel(symbol, horizon=horizon)
    scorer = Scorer(horizon)

    if replay:
        from ml.train_v3 import load_local_klines
        df = load_local_klines(symbol).tail(replay + MIN_TRAIN_BARS).reset_index(drop=True)
        bars = df.to_dict("records")
        logger.info(f"REPLAY on {len(bars)} stored bars (no live API)")
    else:
        logger.info(f"Warming up on {WARMUP_BARS} real bars from Binance...")
        bars = warmup_history(symbol, WARMUP_BARS)
        logger.info(f"Got {len(bars)} bars, last close {bars[-1]['close']}")

    n = 0
    for b in bars:
        model.add_bar(b)
        r = scorer.resolve(b)
        if r:
            n += 1
            print_row(symbol, r, n)
        p = model.predict_next()
        if p:
            scorer.submit(p, b["open_time"] + horizon * 60_000)

    if replay:
        print_summary(symbol, scorer.summary(), model)
        return

    # Live: poll for newly closed bars.
    seen = bars[-1]["open_time"]
    deadline = time.time() + minutes * 60 if minutes else None
    print(f"\nLive. Predicting {horizon} min ahead, retraining every "
          f"{model.retrain_every} bars. Ctrl-C to stop.\n")
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(POLL_SEC)
            try:
                fresh = fetch_klines(symbol, limit=10)
            except Exception as e:
                logger.warning(f"fetch failed: {e}")
                continue
            # Last entry is still forming; only take confirmed closes.
            for b in fresh[:-1]:
                if b["open_time"] <= seen:
                    continue
                seen = b["open_time"]
                refit = model.add_bar(b)
                r = scorer.resolve(b)
                if r:
                    n += 1
                    print_row(symbol, r, n)
                if refit:
                    logger.info(f"refit #{model.n_fits} on {model.last_train_rows} rows")
                p = model.predict_next()
                if p:
                    scorer.submit(p, b["open_time"] + horizon * 60_000)
    except KeyboardInterrupt:
        print("\ninterrupted")
    print_summary(symbol, scorer.summary(), model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--minutes", type=int, default=None, help="stop after N minutes")
    ap.add_argument("--replay", type=int, default=None, help="offline dry run on N stored bars")
    a = ap.parse_args()
    run(a.symbol.upper(), a.minutes, a.replay)
