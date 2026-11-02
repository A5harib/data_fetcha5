"""
Self-checks for model 3. Uses the downloaded daily history, no network.

    python -m model3.test_model3
"""
from __future__ import annotations

import json

import pandas as pd

from model3 import strategy as st
from model3.bot import tick
from model3.data import load


def test_weights_use_only_the_past(c):
    for i in (st.WARMUP + 10, 600, len(c) - 1):
        a = st.target_weights(c.iloc[: i + 1])
        b = st.target_weights(pd.concat([c.iloc[: i + 1], c.iloc[i + 1:] * 3]).iloc[: i + 1])
        assert a == b, f"day {i} weights changed"
        assert sum(a.values()) <= 1 + 1e-9 and len(a) <= st.TOP_K
    print("ok  weights use only the past, never more than 100% invested")


def test_live_matches_backtest(c):
    """Fed one new daily close at a time, with a save and reload in between, the live bot ends where the backtest ends."""
    c = c.iloc[-400:]
    start = c.index[st.WARMUP + 5]
    ref = st.backtest(c, str(start.date()))
    books = st.new_account(100.0, str(start.date()))
    for b in books:   # live buys on day one; the backtest waits for each weekday. Make live wait too.
        b.last_rebalance = "seed"
    acct = {"started": "x", "start_prices": {}, "books": [json.loads(b.dumps()) for b in books]}
    for n in range(c.index.get_loc(start) + 1, len(c) + 1):
        win = c.iloc[max(0, n - st.WARMUP - 30): n]
        acct = json.loads(json.dumps(acct))   # restart from the saved copy every day
        acct, _ = tick(acct, win, {s: float(p) for s, p in win.iloc[-1].items()})
    got = sum(st.Book.from_dict(b).value(c.iloc[-1].to_dict()) for b in acct["books"])
    want = float(st.equity(ref).iloc[-1])
    assert abs(got - want) < 1e-3, f"live {got} vs backtest {want}"
    print(f"ok  live run matches the backtest (${got:.2f})")


def test_restart_never_fills_at_old_prices(c):
    """After 10 missed days, a due sleeve trades once, at today's prices, not at a past close."""
    c = c.iloc[-200:]
    books = st.new_account(100.0, str(c.index[st.WARMUP + 5].date()))
    b = books[0]
    st.step(b, c.iloc[:-10], live=True)
    n = len(b.trades)
    now = {s: float(p) * 1.01 for s, p in c.iloc[-1].items()}
    st.step(b, c, fill=now, live=True)
    new = b.trades[n:]
    assert len(b.rebalances) and b.rebalances[-1]["day"] == c.index[-1].strftime("%Y-%m-%d")
    assert all(t["price"] == now[t["symbol"]] for t in new)
    print("ok  a restart trades only at current prices")


def test_cash_never_negative(c):
    for b in st.backtest(c, "2022-06-01"):
        assert b.cash >= 0 and all(q > 0 for q in b.qty.values())
    print("ok  cash and holdings never go negative")


if __name__ == "__main__":
    c = load()
    test_weights_use_only_the_past(c)
    test_live_matches_backtest(c)
    test_restart_never_fills_at_old_prices(c)
    test_cash_never_negative(c)
