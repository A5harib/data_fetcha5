"""
Hourly spot candles for model 4, from Binance's public spot API
(data-api.binance.vision), which answers from the US-hosted Space. Gold is
PAXGUSDT. Bars are indexed by their open time (UTC).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

import pandas as pd

from forecast.data import SPOT, SPOT_SYMBOL

log = logging.getLogger("model4.data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "XAUUSDT"]
COLS = ["open", "high", "low", "close", "volume", "amount"]


def _klines(sym: str, limit: int, end_ms: int | None = None) -> list:
    url = f"{SPOT}/api/v3/klines?symbol={SPOT_SYMBOL.get(sym, sym)}&interval=1h&limit={limit}"
    if end_ms is not None:
        url += f"&endTime={end_ms}"
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:  # flaky connections: back off and retry
            if i == 3:
                raise
            log.info(f"{sym}: retry after {e}")
            time.sleep(2 * (i + 1))
    return []


def _frame(rows: list) -> pd.DataFrame:
    idx = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
    # amount = quote volume (USDT traded), which is what Kronos calls "amount".
    vals = [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7])] for r in rows]
    return pd.DataFrame(vals, index=idx, columns=COLS)


def candles(sym: str, n: int = 400, end_ms: int | None = None) -> tuple[pd.DataFrame, float]:
    """The last `n` closed hourly bars, and the current price (the open bar's latest close)."""
    now = int(time.time() * 1000)
    rows = _klines(sym, min(n + 1, 1000), end_ms)
    price = float(rows[-1][4])
    closed = [r for r in rows if int(r[6]) < now]
    return _frame(closed[-n:]), price


def history(sym: str, bars: int) -> pd.DataFrame:
    """Up to `bars` closed hourly bars, paging back 1000 at a time. For the quick check in test_model4."""
    out, end = [], None
    while len(out) < bars:
        page = _klines(sym, 1000, end)
        if not page:
            break
        out = page + out
        end = int(page[0][0]) - 1
        if len(page) < 1000:
            break
    now = int(time.time() * 1000)
    return _frame([r for r in out if int(r[6]) < now][-bars:])
