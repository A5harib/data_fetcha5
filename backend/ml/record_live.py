"""
Record real Binance market data to disk for training.

The stored data/*_1m.json snapshots are from 2026-09-04 and are a fixed slice
of history. This pulls fresh bars straight from the Binance futures REST API
and appends them to a JSONL file, deduped by open_time, so the training set
grows with real market data instead of being frozen.

Two modes:
  backfill  one shot, pages backwards from now for N bars
  follow    stays running, appends each newly closed bar

Usage:
    python ml/record_live.py BTCUSDT --backfill 40000
    python ml/record_live.py BTCUSDT --follow
    python ml/record_live.py BTCUSDT --backfill 40000 --follow
"""
import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("record_live")

REST = settings.BINANCE_FUTURES_REST_URL
DATA_DIR = BASE_DIR / "data"
MAX_PER_REQ = 1500
POLL_SEC = 20


def store_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_1m_live.jsonl"


def _get(url: str, retries: int = 4):
    """GET with backoff. The live tracker saw DNS drops, so this must not die on one."""
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i
            logger.warning(f"fetch failed ({e}), retry in {wait}s")
            time.sleep(wait)


def fetch(symbol: str, limit: int = MAX_PER_REQ, end_time: int | None = None) -> list[dict]:
    url = f"{REST}/fapi/v1/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    if end_time:
        url += f"&endTime={end_time}"
    raw = _get(url)
    out = []
    for k in raw:
        vol, buy = float(k[5]), float(k[9])
        out.append({"open_time": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": vol,
                    "buy_volume": buy, "sell_volume": max(0.0, vol - buy),
                    "trades": int(k[8]), "quote_volume": float(k[7])})
    return out


def load_existing(symbol: str) -> dict[int, dict]:
    p = store_path(symbol)
    if not p.exists():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                rows[r["open_time"]] = r
            except json.JSONDecodeError:
                continue
    return rows


def save(symbol: str, rows: dict[int, dict]):
    p = store_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in sorted(rows):
            f.write(json.dumps(rows[t]) + "\n")


def backfill(symbol: str, want: int) -> dict[int, dict]:
    rows = load_existing(symbol)
    logger.info(f"{symbol}: {len(rows)} bars already stored")
    end = None
    while len(rows) < want:
        try:
            chunk = fetch(symbol, MAX_PER_REQ, end)
        except Exception as e:
            # A dropped connection must not throw away everything downloaded
            # so far -- save what we have and stop cleanly.
            logger.warning(f"{symbol}: fetch gave up ({e}); keeping {len(rows)} bars")
            break
        if not chunk:
            break
        new = 0
        for b in chunk:
            if b["open_time"] not in rows:
                rows[b["open_time"]] = b
                new += 1
        end = chunk[0]["open_time"] - 1
        logger.info(f"{symbol}: +{new} (total {len(rows)})")
        # Checkpoint every page so a later failure cannot lose progress.
        save(symbol, rows)
        if new == 0:
            break
        time.sleep(0.3)  # stay well inside Binance rate limits
    save(symbol, rows)
    logger.info(f"{symbol}: saved {len(rows)} bars -> {store_path(symbol).name}")
    return rows


def follow(symbol: str, rows: dict[int, dict] | None = None):
    rows = rows if rows is not None else load_existing(symbol)
    logger.info(f"{symbol}: following live. Ctrl-C to stop. ({len(rows)} bars held)")
    try:
        while True:
            time.sleep(POLL_SEC)
            try:
                fresh = fetch(symbol, 10)
            except Exception as e:
                logger.warning(f"poll failed: {e}")
                continue
            # Drop the last bar: still forming.
            added = 0
            for b in fresh[:-1]:
                if b["open_time"] not in rows:
                    rows[b["open_time"]] = b
                    added += 1
            if added:
                save(symbol, rows)
                last = rows[max(rows)]
                logger.info(f"{symbol}: +{added} bars (total {len(rows)}) close={last['close']}")
    except KeyboardInterrupt:
        save(symbol, rows)
        logger.info(f"{symbol}: stopped, {len(rows)} bars saved")


def demo():
    """Self-check: dedupe and round-trip, no network."""
    import tempfile, os
    global DATA_DIR
    with tempfile.TemporaryDirectory() as td:
        old = DATA_DIR
        DATA_DIR = Path(td)
        try:
            rows = {i: {"open_time": i, "close": float(i)} for i in range(5)}
            save("TEST", rows)
            back = load_existing("TEST")
            assert len(back) == 5 and back[3]["close"] == 3.0, back
            # A re-save with an overlapping batch must not duplicate.
            rows.update({i: {"open_time": i, "close": float(i)} for i in range(3, 8)})
            save("TEST", rows)
            back = load_existing("TEST")
            assert len(back) == 8, len(back)
            assert sorted(back) == list(range(8)), "ordering broken"
            print("record_live demo OK")
        finally:
            DATA_DIR = old


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--backfill", type=int, default=None)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        rows = backfill(a.symbol.upper(), a.backfill) if a.backfill else None
        if a.follow:
            follow(a.symbol.upper(), rows)
        elif not a.backfill:
            ap.error("pass --backfill N and/or --follow")
