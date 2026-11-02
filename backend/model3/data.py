"""
Daily spot closes for model 3, from Binance's public spot API
(data-api.binance.vision), which answers from the US-hosted Space. Gold is
PAXGUSDT, a token backed by one ounce of gold.

Only closed daily bars are returned. Bars are dated by their open (UTC), so
the bar dated Sunday is the one that closes at Monday 00:00 UTC.

    python -m model3.data          # download history to data/model3_daily.pkl
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from forecast.data import DATA_DIR, SPOT, SPOT_SYMBOL
from model3.strategy import SYMBOLS

log = logging.getLogger("model3.data")
HIST = DATA_DIR / "model3_daily.pkl"


def _klines(sym: str, start_ms: int | None = None, limit: int = 1000) -> list:
    url = f"{SPOT}/api/v3/klines?symbol={SPOT_SYMBOL.get(sym, sym)}&interval=1d&limit={limit}"
    if start_ms is not None:
        url += f"&startTime={start_ms}"
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


def _frame(rows: list, now_ms: int) -> pd.Series:
    rows = [r for r in rows if int(r[6]) < now_ms]   # closed bars only
    idx = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
    return pd.Series([float(r[4]) for r in rows], index=idx)


def recent(days: int = 120) -> pd.DataFrame:
    """The last `days` closed daily bars for every symbol, one column each."""
    now = int(time.time() * 1000)
    return pd.DataFrame({s: _frame(_klines(s, limit=days + 1), now) for s in SYMBOLS}).sort_index()


def last_prices() -> dict[str, float]:
    """Current prices (the still-open daily bar's latest close), for showing the account's value."""
    out = {}
    for s in SYMBOLS:
        rows = _klines(s, limit=1)
        if rows:
            out[s] = float(rows[-1][4])
    return out


def history(start: str = "2022-01-01") -> pd.DataFrame:
    now = int(time.time() * 1000)
    cols = {}
    for s in SYMBOLS:
        rows, t = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        while True:
            page = _klines(s, t)
            rows += page
            if len(page) < 1000:
                break
            t = int(page[-1][0]) + 1
        cols[s] = _frame(rows, now)
        log.info(f"{s}: {len(cols[s])} days from {cols[s].index[0].date()}")
    df = pd.DataFrame(cols).sort_index()
    df.to_pickle(HIST)
    return df


def load() -> pd.DataFrame:
    if not HIST.exists():
        raise FileNotFoundError(f"{HIST.name} missing. Run: python -m model3.data")
    return pd.read_pickle(HIST)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = history()
    log.info(f"saved {HIST} ({len(df)} days, last {df.index[-1].date()}, now {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
