"""
Market data for the forecaster, from one source of truth: Binance USD-M futures.

Training reads the public archive (data.binance.vision), serving reads the live
REST API. Both return the same kline columns, and both are assembled by
assemble(), so the model never sees data shaped differently live than it was
trained on.

Output frame: one row per closed 1-minute bar, UTC DatetimeIndex on open time.
    open high low close volume quote_volume count taker_buy_volume
    premium   close of the 1m premium index (perp vs spot index basis)
    funding   last *settled* funding rate as of the bar's close
    filled    True where the bar was missing upstream and was forward-filled

    python -m forecast.data --symbol BTCUSDT --start 2022-01
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger("forecast.data")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE = "https://data.binance.vision/data/futures/um"
REST = "https://fapi.binance.com"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]


# ---------------------------------------------------------------- archive

def _get(url: str, retries: int = 4) -> bytes | None:
    """GET with backoff. None on 404, which the archive uses for 'no such file'."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == retries - 1:
                raise
        time.sleep(2 ** i)
    return None


def _archive_csv(kind: str, symbol: str, period: str, label: str) -> pd.DataFrame | None:
    """One archive file (monthly or daily) as a raw frame, cached on disk as the zip."""
    sub = f"{kind}/{symbol}/1m" if kind in ("klines", "premiumIndexKlines") else f"{kind}/{symbol}"
    name = f"{symbol}-1m-{label}.zip" if "/1m" in sub else f"{symbol}-{kind}-{label}.zip"
    cache = RAW_DIR / kind / symbol / name
    if cache.exists():
        blob = cache.read_bytes()
    else:
        blob = _get(f"{ARCHIVE}/{period}/{sub}/{name}")
        if blob is None:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    # Older files have no header row, newer ones do. Read headerless and drop
    # any row whose first field is not a timestamp.
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    return df


def _months(start: str, end: date):
    y, m = map(int, start.split("-"))
    while (y, m) < (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download_klines(kind: str, symbol: str, start: str) -> pd.DataFrame:
    """Monthly files up to last month, then daily files for this month up to yesterday."""
    today = datetime.now(timezone.utc).date()
    jobs = [("monthly", m) for m in _months(start, today)]
    d = today.replace(day=1)
    while d < today:
        jobs.append(("daily", d.isoformat()))
        d += timedelta(days=1)
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda j: _archive_csv(kind, symbol, *j), jobs))
    missing = [label for (_, label), p in zip(jobs, parts) if p is None]
    if missing:
        # Normal before a symbol listed (XAUUSDT is recent); worth seeing otherwise.
        logger.info(f"{kind} {symbol}: {len(missing)} files absent ({missing[0]} .. {missing[-1]})")
    parts = [p for p in parts if p is not None]
    if not parts:
        raise RuntimeError(f"No {kind} archive data for {symbol} from {start}")
    return pd.concat(parts, ignore_index=True)


def fetch_funding(symbol: str, start_ms: int, end_ms: int | None = None) -> pd.DataFrame:
    """Settled funding rates from REST, paginated. Small: three rows a day."""
    rows, cursor = [], start_ms
    while True:
        url = f"{REST}/fapi/v1/fundingRate?symbol={symbol}&startTime={cursor}&limit=1000"
        if end_ms:
            url += f"&endTime={end_ms}"
        batch = json.loads(_get(url))
        rows += batch
        if len(batch) < 1000:
            break
        cursor = batch[-1]["fundingTime"] + 1
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["t", "funding"])
    return pd.DataFrame({"t": df["fundingTime"].astype("int64"), "funding": df["fundingRate"].astype(float)})


# ---------------------------------------------------------------- live

def _rest_klines(path: str, symbol: str, n: int) -> pd.DataFrame:
    """Last n 1m bars, paging backwards. REST caps one page at 1500."""
    pages, end = [], None
    while n > 0:
        url = f"{REST}{path}?symbol={symbol}&interval=1m&limit={min(n, 1500)}"
        if end is not None:
            url += f"&endTime={end}"
        page = json.loads(_get(url))
        if not page:
            break
        pages.insert(0, page)
        n -= len(page)
        end = int(page[0][0]) - 1
    return pd.DataFrame([r for p in pages for r in p], columns=KLINE_COLS)


def fetch_recent(symbol: str, minutes: int = 4000) -> pd.DataFrame:
    """
    Most recent closed 1m bars from live REST, assembled exactly like training data.
    4000 bars covers the 2880-bar feature warmup with room for the reference vol
    EWMA to converge (weight beyond 4000 bars is ~1e-5).
    """
    # Taken before any request: a minute boundary passing mid-fetch must not let
    # the then-open bar through as if it were closed.
    t_req = int(time.time() * 1000)
    kl = _rest_klines("/fapi/v1/klines", symbol, minutes + 1)
    pr = _rest_klines("/fapi/v1/premiumIndexKlines", symbol, minutes + 1)
    start = int(kl["open_time"].iloc[0]) - 9 * 3600_000  # reach back past one funding settlement
    fu = fetch_funding(symbol, start)
    kl = kl[kl["close_time"].astype("int64") < t_req]
    return assemble(kl, pr, fu)


# ---------------------------------------------------------------- assemble

def assemble(klines: pd.DataFrame, premium: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Raw kline/premium/funding frames -> the canonical 1m frame described at the top."""
    k = klines.copy()
    k["open_time"] = pd.to_numeric(k["open_time"]).astype("int64")
    k = k.drop_duplicates("open_time").set_index("open_time").sort_index()
    k = k[KEEP].astype(float)

    pr = premium.copy()
    pr["open_time"] = pd.to_numeric(pr["open_time"]).astype("int64")
    pr = pr.drop_duplicates("open_time").set_index("open_time").sort_index()
    k["premium"] = pr["close"].astype(float).reindex(k.index)

    # Regular minute grid. Exchange outages leave holes, and every rolling window
    # downstream counts rows, so a hole would silently stretch a "60-minute"
    # window. Fill holes as flat zero-volume bars and flag them.
    grid = pd.RangeIndex(k.index[0], k.index[-1] + 60_000, 60_000)
    filled = ~grid.isin(k.index)
    k = k.reindex(grid)
    k["close"] = k["close"].ffill()
    for c in ("open", "high", "low"):
        k[c] = k[c].fillna(k["close"])
    for c in ("volume", "quote_volume", "count", "taker_buy_volume"):
        k[c] = k[c].fillna(0.0)
    # Bridge a few missing premium bars, but whole missing archive days must not
    # pass as a day of constant basis. Longer holes stay NaN, and the rows whose
    # features touch them drop out of the dataset.
    k["premium"] = k["premium"].ffill(limit=5)
    # The archive also carries outages as flat zero-volume bars rather than
    # holes. Runs longer than 3 minutes get the same flag, so labels spanning
    # them are discarded.
    dead = (k["volume"] == 0) & (k["high"] == k["low"])
    run = dead.groupby((~dead).cumsum()).transform("size").where(dead, 0)
    k["filled"] = filled | (run.to_numpy() > 3)

    # Funding settles at t and is known from t on. A bar uses the latest rate
    # settled at or before its close (open + 60s), never one settled later.
    close_ms = k.index.to_numpy() + 60_000
    f = funding.sort_values("t")
    k["funding"] = pd.merge_asof(
        pd.DataFrame({"t": close_ms}), f[["t", "funding"]], on="t", direction="backward"
    )["funding"].to_numpy()

    k.index = pd.to_datetime(k.index, unit="ms", utc=True)
    k.index.name = "time"
    return k


def build_history(symbol: str, start: str = "2022-01") -> pd.DataFrame:
    kl = download_klines("klines", symbol, start)
    pr = download_klines("premiumIndexKlines", symbol, start)
    first = int(pd.to_numeric(kl["open_time"]).min())
    fu = fetch_funding(symbol, first - 9 * 3600_000)
    df = assemble(kl, pr, fu)
    out = DATA_DIR / f"{symbol}_1m_hist.pkl"
    df.to_pickle(out)
    logger.info(f"{symbol}: {len(df):,} bars {df.index[0]} .. {df.index[-1]}, "
                f"{int(df['filled'].sum()):,} filled, saved {out.name}")
    return df


def load_history(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1m_hist.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing. Run: python -m forecast.data --symbol {symbol}")
    return pd.read_pickle(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2022-01")
    a = ap.parse_args()
    build_history(a.symbol.upper(), a.start)
