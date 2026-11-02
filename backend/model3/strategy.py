"""
Model 3: slow trend rotation over BTC, ETH, SOL, BNB, XRP, DOGE and gold.

Models 1 and 2 made 60-minute calls. They were right more often than wrong,
but each trade earned less than the fee, and with 10+ trades a day the fees
ate the account. Model 3 trades about once a week, so fees barely matter, and
it tries to catch moves of weeks instead of minutes.

The rule, checked once a week at a daily close (00:00 UTC):
1. An asset is "trending" if it is above its 50-day average and up over the
   last 20 days.
2. Hold the TOP_K trending assets with the biggest 20-day rise.
3. Size each so that its recent swings match VOL_TARGET a year: a wild coin
   gets a smaller slice. The rest stays in cash (USDT).
4. When nothing is trending, everything sits in cash.

Which weekday it trades on turned out to swing the backtest from $292 to $837
per $100, which is luck. So the account is split into 7 equal parts
("sleeves"), one per weekday, and each part trades once a week on its day.

The settings below were chosen on 2022-06..2024-12 and then tested once on
2025-01..2026-10, which they never saw. `step()` is the only code path: the
backtest and the live bot both call it, so the live bot trades exactly like
the backtest.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "XAUUSDT"]
LOOKBACK = 20       # days of rise used to rank assets
TREND_SMA = 50      # must be above this daily average to be held
TOP_K = 2           # how many assets to hold at most
VOL_TARGET = 0.4    # yearly swing each slot is scaled to (0.6 earned a bit more but fell further)
VOL_DAYS = 30
SLEEVES = 7         # the account is split in 7 parts, each rebalancing on a different weekday
FEE = 0.001             # 0.1% per side, Binance spot taker fee without the BNB discount
MIN_TRADE = 0.01        # skip a trade smaller than 1% of the account, unless it closes a position
WARMUP = max(LOOKBACK, TREND_SMA, VOL_DAYS) + 1


def target_weights(closes: pd.DataFrame) -> dict[str, float]:
    """Share of the account to put in each asset, from daily closes up to and including today."""
    c = closes.iloc[-WARMUP:]
    w = {}
    cand = []
    for sym in c.columns:
        s = c[sym].dropna()
        if len(s) < WARMUP:
            continue
        mom = s.iloc[-1] / s.iloc[-1 - LOOKBACK] - 1
        if s.iloc[-1] > s.iloc[-TREND_SMA:].mean() and mom > 0:
            cand.append((mom, sym, s))
    for mom, sym, s in sorted(cand, reverse=True)[:TOP_K]:
        vol = np.log(s).diff().iloc[-VOL_DAYS:].std() * math.sqrt(365)
        w[sym] = min(1.0, VOL_TARGET / vol) / TOP_K if vol > 0 else 1.0 / TOP_K
    return w


def explain(closes: pd.DataFrame) -> list[dict]:
    """Each asset's numbers behind the decision, for the page."""
    c = closes.iloc[-WARMUP:]
    out = []
    for sym in c.columns:
        s = c[sym].dropna()
        if len(s) < WARMUP:
            continue
        sma = s.iloc[-TREND_SMA:].mean()
        out.append({"symbol": sym, "price": float(s.iloc[-1]),
                    "rise_pct": round(100 * (s.iloc[-1] / s.iloc[-1 - LOOKBACK] - 1), 2),
                    "vs_avg_pct": round(100 * (s.iloc[-1] / sma - 1), 2),
                    "yearly_swing_pct": round(100 * np.log(s).diff().iloc[-VOL_DAYS:].std() * math.sqrt(365), 1)})
    return sorted(out, key=lambda r: -r["rise_pct"])


@dataclass
class Book:
    """The paper account: cash, coins held, and its history."""
    cash: float = 100.0
    start_cash: float = 100.0
    qty: dict = field(default_factory=dict)
    last_day: str | None = None          # last daily bar processed, YYYY-MM-DD
    last_rebalance: str | None = None
    started: str | None = None
    fees_paid: float = 0.0
    trades: list = field(default_factory=list)
    rebalances: list = field(default_factory=list)
    equity: list = field(default_factory=list)   # [day, value] at each daily close
    weekday: int = 6                             # day of the daily bar it rebalances on, Monday=0

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.qty.items() if q)

    def rebalance(self, weights: dict[str, float], prices: dict[str, float], day: str):
        eq = self.value(prices)
        held = {s: q * prices[s] for s, q in self.qty.items() if q}
        diffs = {}
        for s in set(held) | set(weights):
            d = weights.get(s, 0.0) * eq - held.get(s, 0.0)
            closing = weights.get(s, 0.0) == 0.0 and s in held
            if closing or abs(d) >= MIN_TRADE * eq:
                diffs[s] = d
        # Sells first, so the cash is there for the buys.
        for s, d in sorted(diffs.items(), key=lambda kv: kv[1]):
            if d < 0:
                usd = min(-d, held[s])
                q = self.qty[s] if weights.get(s, 0.0) == 0.0 else usd / prices[s]
                usd = q * prices[s]
                self.qty[s] -= q
                if self.qty[s] <= 1e-12:
                    del self.qty[s]
                fee = usd * FEE
                self.cash += usd - fee
                self.fees_paid += fee
                self.trades.append({"day": day, "symbol": s, "side": "sell", "usd": round(usd, 4), "price": prices[s]})
        buys = {s: d for s, d in diffs.items() if d > 0}
        need = sum(buys.values()) * (1 + FEE)
        scale = min(1.0, self.cash / need) if need > 0 else 1.0
        for s, d in buys.items():
            usd = d * scale
            fee = usd * FEE
            self.qty[s] = self.qty.get(s, 0.0) + usd / prices[s]
            self.cash -= usd + fee
            self.fees_paid += fee
            self.trades.append({"day": day, "symbol": s, "side": "buy", "usd": round(usd, 4), "price": prices[s]})
        self.cash = max(self.cash, 0.0)
        self.last_rebalance = day
        self.rebalances.append({"day": day, "value": round(eq, 4), "weights": {s: round(w, 4) for s, w in weights.items()}})

    def dumps(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "Book":
        return cls(**d)


def step(book: Book, closes: pd.DataFrame, fill: dict[str, float] | None = None, live: bool = False) -> bool:
    """
    Move the book through every daily close it has not seen yet, rebalancing
    on each close that falls on its weekday. Returns True if it traded.

    live=True is for the running bot: it never fills at an old price. Missed
    days after a restart only update the value, and if the sleeve's weekday was
    among them (or it has never traded) it rebalances on the newest bar at the
    `fill` prices, which are the prices right now.
    """
    traded = False
    days = closes.index
    todo = [i for i, t in enumerate(days) if book.last_day is None or t.strftime("%Y-%m-%d") > book.last_day]
    missed = False
    for i in todo:
        if i + 1 < WARMUP:
            continue
        t = days[i]
        day = t.strftime("%Y-%m-%d")
        prices = {s: float(p) for s, p in closes.iloc[i].items() if pd.notna(p)}
        last = i == len(days) - 1
        due = t.weekday() == book.weekday
        if live and not last:
            missed |= due
            due = False
        elif live:
            due = due or missed or book.last_rebalance is None
        if due:
            px = dict(prices, **(fill or {})) if live else prices
            book.rebalance(target_weights(closes.iloc[: i + 1]), px, day)
            traded = True
        book.equity.append([day, round(book.value(prices), 4)])
        book.last_day = day
    return traded


def new_account(cash: float = 100.0, start: str | None = None) -> list[Book]:
    """The 7 sleeves. Trading starts on `start`; earlier bars only warm up the averages."""
    books = [Book(cash=cash / SLEEVES, start_cash=cash / SLEEVES, weekday=wd) for wd in range(SLEEVES)]
    if start:
        for b in books:
            b.last_day = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return books


def backtest(closes: pd.DataFrame, start: str | None = None, cash: float = 100.0) -> list[Book]:
    books = new_account(cash, start)
    for b in books:
        step(b, closes)
    return books


def equity(books: list[Book]) -> pd.Series:
    """The whole account's value at each daily close."""
    return sum(pd.Series({pd.Timestamp(d): v for d, v in b.equity}) for b in books)


def yearly(books: list[Book]) -> dict[str, dict]:
    """By-year return, worst drop and % of months up, from the account's daily values."""
    e = equity(books)
    out = {}
    prev = sum(b.start_cash for b in books)
    for y, s in e.groupby(e.index.year):
        m = s.resample("ME").last()
        mret = m.div(m.shift(1).fillna(prev)) - 1
        out[str(y)] = {"ret_pct": round(100 * (s.iloc[-1] / prev - 1), 1),
                       "worst_drop_pct": round(100 * (s / s.cummax().clip(lower=prev) - 1).min(), 1),
                       "months_up": f"{int((mret > 0).sum())}/{len(mret)}"}
        prev = s.iloc[-1]
    return out
