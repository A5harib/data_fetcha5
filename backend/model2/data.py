"""
Spot price history for model 2.

The bots on the Hugging Face Space read Binance spot prices (futures is blocked
from US servers), so model 2 learns from spot history too. Gold is PAXGUSDT, a
token backed by one ounce of gold, which is what the Space sees live.

The frame is built by forecast.data.assemble with zero premium and funding,
exactly as forecast.data.LiveFeed(source="spot") builds it live, so training
and live data have the same shape.

    python -m model2.data --symbol BTCUSDT --start 2022-01
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from forecast.data import DATA_DIR, KLINE_COLS, RAW_DIR, SPOT_SYMBOL, _get, _months, assemble

log = logging.getLogger("model2.data")
ARCHIVE = "https://data.binance.vision/data/spot"


def _file(sym: str, period: str, label: str) -> pd.DataFrame | None:
    name = f"{sym}-1m-{label}.zip"
    cache = RAW_DIR / "spot_klines" / sym / name
    if cache.exists():
        blob = cache.read_bytes()
    else:
        blob = _get(f"{ARCHIVE}/{period}/klines/{sym}/1m/{name}")
        if blob is None:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()].copy()
    # Spot files from 2025 on stamp times in microseconds, older ones in ms.
    for c in ("open_time", "close_time"):
        t = pd.to_numeric(df[c]).astype("int64")
        df[c] = t.where(t < 1e14, t // 1000)
    return df


def spot_frame(kl: pd.DataFrame) -> pd.DataFrame:
    """Raw spot klines -> canonical 1m frame, the same way LiveFeed(source='spot') does it."""
    pr = kl.copy()
    pr["close"] = 0.0
    return assemble(kl, pr, pd.DataFrame({"t": [0], "funding": [0.0]}))


def build(symbol: str, start: str = "2022-01") -> pd.DataFrame:
    sym = SPOT_SYMBOL.get(symbol, symbol)
    today = datetime.now(timezone.utc).date()
    jobs = [("monthly", m) for m in _months(start, today)]
    d = today.replace(day=1)
    while d < today:
        jobs.append(("daily", d.isoformat()))
        d += timedelta(days=1)
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda j: _file(sym, *j), jobs) if p is not None]
    df = spot_frame(pd.concat(parts, ignore_index=True))
    out = DATA_DIR / f"{symbol}_spot_1m_hist.pkl"
    df.to_pickle(out)
    log.info(f"{symbol} ({sym} spot): {len(df):,} bars {df.index[0]} .. {df.index[-1]}, saved {out.name}")
    return df


def load(symbol: str) -> pd.DataFrame:
    p = DATA_DIR / f"{symbol}_spot_1m_hist.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p.name} missing. Run: python -m model2.data --symbol {symbol}")
    return pd.read_pickle(p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2022-01")
    a = ap.parse_args()
    build(a.symbol.upper(), a.start)
