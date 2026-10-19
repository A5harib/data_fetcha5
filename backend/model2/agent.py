"""
Model 2: a trading agent that keeps learning from the market while it runs.

How it decides, every closed 1-minute bar:

  1. Score. A linear model turns ~38 price, volume and trend features into a
     score: positive means "likely up over the next hour", negative "likely
     down". Its weights are fit by recursive least squares on the sign of the
     move that followed, with older hours slowly forgotten (half-life 90 days),
     so it follows the market as it changes.
  2. Learn. Once an hour it adds every minute whose hour-ahead outcome is now
     known and refits. Nothing it learns from is newer than the bar it acts on.
  3. Pickiness levels. Four virtual traders run side by side. Each enters when
     the score is in the top 5%, 1%, 0.5% or 0.1% of the last 30 days, holds
     while the score still agrees with the position (at least 30 minutes, at
     most 6 hours), and exits when it turns. Each one's trades are booked after
     fees.
  4. Self-tuning. The real bot follows whichever level made the most money,
     after fees, on trades closed in the last 30 days. If none made money, it
     sits out. Its own results feed this choice, so it trades more when its
     style is working and stops when it is not.

Training replays years of history through exactly this loop (model2.train),
so the saved state is what the agent had learned by the end of history. Live,
the same object continues minute by minute (model2.bot).
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np
import pandas as pd

from forecast.features import FEATURES, build_features, reference_vol

H = 60              # minutes ahead the score predicts, and the learning label
FEE = 0.0006        # per side, same as the model 1 paper bots
HALF_LIFE_DAYS = 90
RIDGE = 1000.0
LEVELS = (0.95, 0.99, 0.995, 0.999)   # enter on the top 5% / 1% / 0.5% / 0.1% of scores
WINDOW_HOURS = 720  # 30 days of hourly score samples set each level's bar
MIN_SAMPLES = 168   # a level does not trade until it has 7 days of samples
MIN_HOLD, MAX_HOLD = 30, 360
LOOKBACK_MIN = 30 * 1440  # self-tuning judges levels on the last 30 days
WARMUP = 43200      # longest feature lookback (30-day trend), in bars
# Sit out when no style made money lately. MODEL2_SIT_OUT=0 makes the paper
# account always follow the least-bad style instead.
SIT_OUT = os.environ.get("MODEL2_SIT_OUT", "1") != "0"

# Spot data has no perp premium or funding, and the funding clock means nothing.
_DROP = {"premium", "premium_dev_1440", "premium_chg_60", "funding", "min_to_funding"}
BASE = [f for f in FEATURES if f not in _DROP]
EXTRA = ["tr_4320", "tr_10080", "tr_20160", "tr_43200", "ma_10080", "ma_43200", "dvol_7d"]
COLS = BASE + EXTRA


def features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Model 1's past-only features plus multi-day trend and vol regime."""
    X = build_features(df, symbol)[BASE].astype("float64")
    lc = np.log(df["close"])
    sig = reference_vol(df)
    for k in (4320, 10080, 20160, 43200):
        X[f"tr_{k}"] = (lc - lc.shift(k)) / (sig * np.sqrt(k))
    for k in (10080, 43200):
        X[f"ma_{k}"] = (lc - lc.rolling(k).mean()) / (sig * np.sqrt(k))
    X["dvol_7d"] = np.log(sig / sig.rolling(10080).mean())
    return X[COLS]


def labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sign of the move over the next H bars, and whether that window is clean."""
    c = df["close"].to_numpy()
    fwd = np.full(len(c), np.nan)
    fwd[:-H] = np.log(c[H:] / c[:-H])
    filled_ahead = pd.Series(df["filled"].to_numpy(float)).rolling(H).sum().shift(-H).to_numpy()
    return np.sign(np.nan_to_num(fwd)), np.isfinite(fwd) & (filled_ahead == 0)


def minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar open times as whole minutes since 1970. Unit-safe: pandas 3 may store ms or us."""
    return (index.as_unit("s").asi8 // 60).astype(np.int64)


class Agent:
    def __init__(self, symbol: str, scaler: dict):
        self.symbol = symbol
        self.scaler = {k: np.asarray(v, float) for k, v in scaler.items()}
        d = len(COLS) + 1
        self.S = np.eye(d) * RIDGE
        self.b = np.zeros(d)
        self.w = np.zeros(d)
        self.learned_up_to = None   # minute: rows before this have been learned from
        self.hours_learned = 0
        self.samples: deque = deque(maxlen=WINDOW_HOURS)
        self.thr = {q: np.inf for q in LEVELS}
        self.levels = {q: {"pos": 0, "held": 0, "entry": None, "entry_t": None, "trades": []} for q in LEVELS}
        self.last_bar = None        # minute of the last bar processed
        self.score = None           # score of the last bar
        self.real = None            # the money-tracking bot, set by start_real()

    # ------------------------------------------------------------ inputs

    def z(self, X: np.ndarray) -> np.ndarray:
        """Winsorize and standardize with the scaler fit on the first months of history; add bias."""
        s = self.scaler
        Z = (np.clip(X, s["lo"], s["hi"]) - s["mu"]) / s["sd"]
        Z = np.nan_to_num(Z)
        return np.c_[Z, np.ones(len(Z))] if Z.ndim == 2 else np.r_[Z, 1.0]

    @staticmethod
    def fit_scaler(X: np.ndarray) -> dict:
        lo, hi = np.quantile(X, 0.005, axis=0), np.quantile(X, 0.995, axis=0)
        C = np.clip(X, lo, hi)
        return {"lo": lo, "hi": hi, "mu": C.mean(0), "sd": C.std(0) + 1e-9}

    # ------------------------------------------------------------ learning

    def learn(self, Z: np.ndarray, y: np.ndarray):
        """One hour's update: forget a little, add the newly resolved rows, refit."""
        lam = (1 - np.log(2) / (HALF_LIFE_DAYS * 1440)) ** 60
        d = len(self.w)
        self.S = self.S * lam + Z.T @ Z + np.eye(d) * RIDGE * (1 - lam)
        self.b = self.b * lam + Z.T @ y
        self.w = np.linalg.solve(self.S, self.b)
        self.hours_learned += 1

    # ------------------------------------------------------------ acting

    def best_level(self, now: int):
        """Level with the most money made after fees on trades closed in the last 30 days, or None."""
        best, best_v = None, (0.0 if SIT_OUT else -np.inf)
        for q, L in self.levels.items():
            v = sum(n for t, n in L["trades"] if now - LOOKBACK_MIN <= t < now)
            if v > best_v:
                best, best_v = q, v
        return best

    def step(self, t: int, z: np.ndarray, price: float) -> dict:
        """Process the bar that opened at minute t and just closed at `price`. Returns events."""
        ev = {}
        s = float(z @ self.w)
        if t % 60 == 0:
            n = len(self.samples)
            if n >= MIN_SAMPLES:
                a = np.asarray(self.samples)
                self.thr = {q: float(np.quantile(a, q)) for q in LEVELS}
            self.samples.append(abs(s))

        choice = self.best_level(t) if self.real is not None and self.real["position"] is None else None

        for q, L in self.levels.items():
            if L["pos"]:
                L["held"] += 1
                if L["held"] >= MIN_HOLD and (s * L["pos"] < 0 or L["held"] >= MAX_HOLD):
                    gross = L["pos"] * np.log(price / L["entry"])
                    L["trades"].append((t, float(gross - 2 * FEE)))
                    if self.real is not None and L.get("entry_t", 0) >= self.real.get("started_min", 0):
                        lv = L.setdefault("live", {"n": 0, "right": 0, "growth": 1.0})
                        lv["n"] += 1
                        simple = L["pos"] * (price / L["entry"] - 1)  # same booking as the paper account
                        lv["right"] += int(simple > 0)
                        lv["growth"] *= (1 - FEE) ** 2 * (1 + simple)
                    if self.real and self.real["position"] and self.real["position"]["level"] == q:
                        ev["closed"] = self._close_real(t, price)
                    L["pos"] = 0
            if L["pos"] == 0 and abs(s) > self.thr[q]:
                L.update(pos=1 if s > 0 else -1, held=0, entry=price, entry_t=t)
                if self.real is not None and self.real["position"] is None and choice == q:
                    ev["opened"] = self._open_real(t, price, L["pos"], q, s)
            # keep 30 days of closed trades, enough for the self-tuning window
            while L["trades"] and L["trades"][0][0] < t - LOOKBACK_MIN:
                L["trades"].pop(0)

        self.last_bar = t
        self.score = s
        return ev

    # ------------------------------------------------------------ real money (paper)

    def start_real(self, cash: float, started: str):
        # started_min: styles count their calls toward the live record from this bar on.
        self.real = {"started": started, "started_min": (self.last_bar or 0) + 1, "start_cash": cash,
                     "cash": cash, "cash_no_fees": cash, "position": None, "trades": []}

    def _open_real(self, t, price, side, q, s):
        self.real["position"] = {"side": side, "entry": price, "entry_t": t, "level": q, "score": round(s, 4)}
        return self.real["position"]

    def _close_real(self, t, price):
        p, r = self.real["position"], self.real
        gross = price / p["entry"] - 1
        gross = gross if p["side"] > 0 else -gross
        r["cash"] *= (1 - FEE) * (1 + gross) * (1 - FEE)
        r["cash_no_fees"] *= 1 + gross
        tr = {**p, "exit": price, "exit_t": t, "move_pct": round(100 * gross, 4), "right": gross > 0,
              "cash": round(r["cash"], 4), "cash_no_fees": round(r["cash_no_fees"], 4)}
        r["trades"].append(tr)
        r["position"] = None
        return tr

    # ------------------------------------------------------------ save / load

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "cols": COLS,
            "scaler": {k: v.tolist() for k, v in self.scaler.items()},
            "S": self.S.tolist(), "b": self.b.tolist(), "w": self.w.tolist(),
            "learned_up_to": self.learned_up_to, "hours_learned": self.hours_learned,
            "samples": list(self.samples), "thr": {str(q): v for q, v in self.thr.items()},
            "levels": {str(q): L for q, L in self.levels.items()},
            "last_bar": self.last_bar, "score": self.score, "real": self.real,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Agent":
        if d["cols"] != COLS:
            raise ValueError("saved agent was built on different features")
        a = cls(d["symbol"], d["scaler"])
        a.S, a.b, a.w = np.asarray(d["S"]), np.asarray(d["b"]), np.asarray(d["w"])
        a.learned_up_to, a.hours_learned = d["learned_up_to"], d["hours_learned"]
        a.samples = deque(d["samples"], maxlen=WINDOW_HOURS)
        a.thr = {float(q): v for q, v in d["thr"].items()}
        a.levels = {float(q): {**L, "trades": [tuple(x) for x in L["trades"]]} for q, L in d["levels"].items()}
        a.last_bar, a.real, a.score = d["last_bar"], d["real"], d.get("score")
        return a

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), default=float)


def advance(agent: Agent, df: pd.DataFrame, X: pd.DataFrame | None = None, on_event=None):
    """
    Feed every closed bar of `df` newer than agent.last_bar through the agent,
    learning once an hour from rows whose outcome is known. Used by both the
    history replay and the live loop, so they cannot drift apart.
    """
    if X is None:
        X = features(df, agent.symbol)
    t_min = minutes(df.index)
    A = X.to_numpy(float)
    ok = np.isfinite(A).all(axis=1)
    y, clean = labels(df)
    close = df["close"].to_numpy()

    first = 0 if agent.last_bar is None else int(np.searchsorted(t_min, agent.last_bar, side="right"))
    Z = agent.z(np.nan_to_num(A))
    for i in range(first, len(df)):
        t = int(t_min[i])
        if not ok[i]:
            continue
        if t % 60 == 0:
            # Learn from rows j whose H-bar outcome closed by the end of the previous bar.
            lo = agent.learned_up_to
            hi_t = t - 1 - H  # last learnable row's minute
            if lo is None:
                lo = hi_t - 60 + 1
            j0 = int(np.searchsorted(t_min, lo, side="left"))
            j1 = int(np.searchsorted(t_min, hi_t, side="right"))
            if j1 > j0:
                m = np.arange(j0, j1)
                m = m[ok[m] & clean[m]]
                if len(m):
                    agent.learn(Z[m], y[m])
            agent.learned_up_to = hi_t + 1
        ev = agent.step(t, Z[i], float(close[i]))
        if ev and on_event:
            on_event(ev, df.index[i])
    return agent
