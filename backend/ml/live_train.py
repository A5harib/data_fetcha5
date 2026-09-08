"""
Record real Binance bars and retrain the reversal classifier in the same loop.

One process does three things per newly closed bar:
  1. appends the bar to data/<SYM>_1m_live.jsonl (same store as record_live.py)
  2. scores the bar with the current model and parks the call until its
     3-bar outcome is known, then reports whether it was right
  3. every RETRAIN_EVERY bars, refits on the whole accumulated store

Refitting is honest: the label needs HORIZON bars of future, so the newest
HORIZON bars are never trained on, and the calibrator + decision threshold are
fitted on a held-out tail rather than on the training rows.

This is genuinely cheap. ~17k gated rows and 12 features refit in about a
second, against a 60-second bar, so training keeps up with recording easily.

Usage:
    python ml/live_train.py BTCUSDT                 # runs until Ctrl-C
    python ml/live_train.py BTCUSDT --minutes 120
    python ml/live_train.py BTCUSDT --replay 3000   # offline, no network
"""
import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from config import settings
from ml.features_v3 import build_dataset, build_frame, FEATURE_NAMES_V3, HORIZON
from ml.train_v3 import _fit_one
from ml.record_live import fetch, load_existing, save, store_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_train")

RETRAIN_EVERY = 60      # closed bars between refits
MIN_ROWS = 800          # gated rows required before the model is usable
POLL_SEC = 20


class LiveTrainer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.gate = settings.gate_for(self.symbol)
        self.thr_pct = settings.reversal_threshold_for(self.symbol)
        self.bars: dict[int, dict] = load_existing(self.symbol)
        self.model = None
        self.iso = None
        self.threshold = 0.5
        self.bars_since_fit = 0
        self.n_fits = 0
        self.train_rows = 0
        self.oos_auc = float("nan")
        # Open calls awaiting their outcome, and the resolved scorecard.
        self.pending: deque = deque()
        self.calls = 0
        self.hits = 0
        self.fired = 0
        self.fired_hits = 0

    # ---------- training ----------

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.bars[t] for t in sorted(self.bars)])

    def refit(self) -> bool:
        df = self.frame()
        if len(df) < MIN_ROWS:
            return False
        X, y, _ = build_dataset(df, self.thr_pct, gated=True, gate_pct=self.gate)
        if len(X) < MIN_ROWS or y.sum() < 60:
            return False

        # Hold out the newest 15% to measure out-of-sample AUC honestly.
        cut = int(len(X) * 0.85)
        Xtr, ytr = X.iloc[:cut].values, y.iloc[:cut].values
        Xte, yte = X.iloc[cut:].values, y.iloc[cut:].values
        if ytr.sum() < 30 or yte.sum() < 5:
            return False

        clf, iso, _, p_te, thr = _fit_one(Xtr, ytr, Xte, yte)
        from sklearn.metrics import roc_auc_score
        try:
            self.oos_auc = float(roc_auc_score(yte, p_te))
        except ValueError:
            self.oos_auc = float("nan")

        self.model, self.iso, self.threshold = clf, iso, thr
        self.n_fits += 1
        self.train_rows = len(X)
        self.bars_since_fit = 0
        return True

    # ---------- prediction ----------

    def predict_latest(self) -> dict | None:
        """Score the newest bar. None when it is not a reversal setup."""
        if self.model is None:
            return None
        df = self.frame().tail(300)
        if len(df) < 140:
            return None
        past3 = float((df["close"].iloc[-1] / df["close"].iloc[-4] - 1) * 100)
        if abs(past3) < self.gate:
            return None
        row = build_frame(df)[FEATURE_NAMES_V3].iloc[-1:]
        if row.isna().any().any():
            return None
        raw = float(self.model.predict_proba(row.values.astype(np.float32))[0, 1])
        prob = float(self.iso.predict([raw])[0])
        return {
            "open_time": int(df["open_time"].iloc[-1]),
            "close": float(df["close"].iloc[-1]),
            "prob": prob,
            "fired": prob >= self.threshold,
            "direction": "BEARISH" if past3 > 0 else "BULLISH",
            "past3": past3,
        }

    def submit(self, call: dict):
        self.pending.append(call)

    def resolve(self):
        """Score any call whose 3-bar outcome is now known."""
        done = []
        times = sorted(self.bars)
        if not times:
            return done
        newest = times[-1]
        while self.pending:
            c = self.pending[0]
            target = c["open_time"] + HORIZON * 60_000
            if newest < target:
                break
            self.pending.popleft()
            window = [self.bars[t] for t in times
                      if c["open_time"] < t <= target]
            if len(window) < HORIZON:
                continue
            base = c["close"]
            hi = max(w["high"] for w in window)
            lo = min(w["low"] for w in window)
            if c["direction"] == "BEARISH":
                hit = (lo - base) / base * 100 <= -self.thr_pct
            else:
                hit = (hi - base) / base * 100 >= self.thr_pct
            self.calls += 1
            self.hits += int(hit)
            if c["fired"]:
                self.fired += 1
                self.fired_hits += int(hit)
            done.append({**c, "hit": hit})
        return done

    def scorecard(self) -> str:
        base = self.hits / self.calls if self.calls else float("nan")
        prec = self.fired_hits / self.fired if self.fired else float("nan")
        return (f"setups {self.calls}  base {base:.1%}  |  "
                f"fired {self.fired}  precision {prec:.1%}  |  "
                f"fits {self.n_fits} rows {self.train_rows} oosAUC {self.oos_auc:.3f}")

    # ---------- loop ----------

    def ingest(self, bar: dict) -> bool:
        if bar["open_time"] in self.bars:
            return False
        self.bars[bar["open_time"]] = bar
        self.bars_since_fit += 1
        return True


def run(symbol: str, minutes: int | None, replay: int | None):
    t = LiveTrainer(symbol)

    if replay:
        from ml.train_v3 import load_local_klines
        df = load_local_klines(symbol).tail(replay).reset_index(drop=True)
        stream = df.to_dict("records")
        t.bars = {}
        logger.info(f"REPLAY {len(stream)} stored bars")
        for b in stream:
            t.ingest(b)
            for r in t.resolve():
                mark = "HIT " if r["hit"] else "miss"
                if r["fired"]:
                    print(f"  FIRED {r['direction']:<7} p={r['prob']:.3f} -> {mark}")
            if t.bars_since_fit >= RETRAIN_EVERY and t.refit():
                logger.info(f"refit #{t.n_fits}: {t.scorecard()}")
            c = t.predict_latest()
            if c:
                t.submit(c)
        print("\n" + "=" * 72)
        print(f"{symbol} REPLAY  {t.scorecard()}")
        print("=" * 72)
        return

    if len(t.bars) < MIN_ROWS:
        logger.info(f"only {len(t.bars)} bars stored; pulling history first")
        from ml.record_live import backfill
        t.bars = backfill(symbol, 20000)

    t.refit()
    logger.info(f"initial fit: {t.scorecard()}")
    logger.info(f"following live, retrain every {RETRAIN_EVERY} bars. Ctrl-C to stop.")

    deadline = time.time() + minutes * 60 if minutes else None
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(POLL_SEC)
            try:
                fresh = fetch(symbol, 10)
            except Exception as e:
                logger.warning(f"poll failed: {e}")
                continue
            added = 0
            for b in fresh[:-1]:          # last bar still forming
                if t.ingest(b):
                    added += 1
            if not added:
                continue
            save(symbol, t.bars)

            for r in t.resolve():
                mark = "HIT " if r["hit"] else "miss"
                tag = "FIRED" if r["fired"] else "     "
                print(f"{tag} {r['direction']:<7} p={r['prob']:.3f} -> {mark}  "
                      f"({t.scorecard()})")

            if t.bars_since_fit >= RETRAIN_EVERY:
                if t.refit():
                    logger.info(f"refit #{t.n_fits}: {t.scorecard()}")

            c = t.predict_latest()
            if c:
                t.submit(c)
                if c["fired"]:
                    print(f"  new call: {c['direction']} p={c['prob']:.3f} "
                          f"thr={t.threshold:.2f} @ {c['close']}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    save(symbol, t.bars)
    print("\n" + "=" * 72)
    print(f"{symbol}  {t.scorecard()}")
    print(f"stored {len(t.bars)} bars -> {store_path(symbol).name}")
    print("=" * 72)


def demo():
    """Self-check: outcome resolution logic, no network."""
    t = LiveTrainer("TEST")
    t.thr_pct = 0.5
    base = 100.0
    # Bar 0 is the call; bars 1..3 are the outcome window.
    t.bars = {i * 60000: {"open_time": i * 60000, "open": base, "high": base,
                          "low": base, "close": base, "volume": 1.0,
                          "buy_volume": .5, "sell_volume": .5} for i in range(4)}
    # A bearish call resolves as a hit only if price drops >= thr_pct.
    t.bars[3 * 60000]["low"] = base * 0.99
    t.submit({"open_time": 0, "close": base, "prob": 0.9, "fired": True,
              "direction": "BEARISH", "past3": 0.3})
    out = t.resolve()
    assert len(out) == 1 and out[0]["hit"] is True, out
    assert t.calls == 1 and t.fired == 1 and t.fired_hits == 1

    # Same setup without the drop must miss.
    t2 = LiveTrainer("TEST"); t2.thr_pct = 0.5
    t2.bars = {i * 60000: {"open_time": i * 60000, "open": base, "high": base,
                           "low": base, "close": base, "volume": 1.0,
                           "buy_volume": .5, "sell_volume": .5} for i in range(4)}
    t2.submit({"open_time": 0, "close": base, "prob": 0.9, "fired": True,
               "direction": "BEARISH", "past3": 0.3})
    out = t2.resolve()
    assert len(out) == 1 and out[0]["hit"] is False, out

    # A call whose window has not closed yet must stay pending.
    t3 = LiveTrainer("TEST"); t3.thr_pct = 0.5
    t3.bars = {0: {"open_time": 0, "high": base, "low": base, "close": base}}
    t3.submit({"open_time": 0, "close": base, "prob": .9, "fired": True,
               "direction": "BULLISH", "past3": -0.3})
    assert t3.resolve() == [] and len(t3.pending) == 1
    print("live_train demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--replay", type=int, default=None)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        run(a.symbol.upper(), a.minutes, a.replay)
