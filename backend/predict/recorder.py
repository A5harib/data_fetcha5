"""
Records live market state to disk, one snapshot per second.

Historical klines carry no order book, so a model that uses book features
cannot be trained from downloaded history. It has to be recorded. This runs
alongside the live stream and writes JSONL that train.py consumes.

Run standalone:
    python -m predict.recorder --symbol BTCUSDT --hours 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from predict.features import build_snapshot

logger = logging.getLogger("recorder")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# 120s of history is the deepest window any feature needs. Keep a little
# slack so a burst of trades cannot evict data a feature still wants.
HISTORY_MS = 180_000


class MarketRecorder:
    """
    Holds trailing market state and emits one snapshot per second.

    Fed by whatever transport you like: call on_trade() and on_depth() as
    data arrives, then tick() once a second. Keeping the transport out of
    this class is what lets the same code serve the recorder, the live
    predictor, and the tests.
    """

    def __init__(self, symbol: str, out_path: Optional[Path] = None):
        self.symbol = symbol.upper()
        self.out_path = out_path
        self._fh = None

        self.mid_history: Deque[tuple] = deque()
        self.trade_history: Deque[tuple] = deque()
        self.imbalance_history: Deque[tuple] = deque()

        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.last_trade_ms = 0
        self.snapshots_written = 0
        self._last_tick_sec = 0

    def open(self):
        if self.out_path and self._fh is None:
            self._fh = self.out_path.open("a", encoding="utf-8")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def on_trade(self, t_ms: int, price: float, qty: float, is_buyer_maker: bool):
        """A taker buy is +qty, a taker sell is -qty, matching the existing buffer's convention."""
        signed = -qty if is_buyer_maker else qty
        self.trade_history.append((t_ms, signed, qty))
        self.last_trade_ms = max(self.last_trade_ms, t_ms)
        self._prune(t_ms)

    def on_depth(self, t_ms: int, bids: list, asks: list, levels: int = 5):
        """Top-N book. bids/asks are [[price, qty], ...] as strings or floats."""
        if not bids or not asks:
            return
        b = [(float(p), float(q)) for p, q in bids[:levels]]
        a = [(float(p), float(q)) for p, q in asks[:levels]]
        self.best_bid = b[0][0]
        self.best_ask = a[0][0]
        self.bid_qty = sum(q for _, q in b)
        self.ask_qty = sum(q for _, q in a)

        total = self.bid_qty + self.ask_qty
        imb = (self.bid_qty - self.ask_qty) / total if total > 0 else 0.0
        self.imbalance_history.append((t_ms, imb))

        mid = (self.best_bid + self.best_ask) / 2.0
        if mid > 0:
            self.mid_history.append((t_ms, mid))
        self._prune(t_ms)

    def _prune(self, now_ms: int):
        cutoff = now_ms - HISTORY_MS
        for dq in (self.mid_history, self.trade_history, self.imbalance_history):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def tick(self, now_ms: Optional[int] = None):
        """
        Emit one snapshot if a new second has started. Returns the Snapshot,
        or None while the market state is still too thin to describe honestly.
        """
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        sec = now_ms // 1000
        if sec == self._last_tick_sec:
            return None
        self._last_tick_sec = sec

        if not self.mid_history or self.best_bid <= 0 or self.best_ask <= 0:
            return None

        mid = (self.best_bid + self.best_ask) / 2.0
        snap = build_snapshot(
            t_ms=now_ms,
            mid=mid,
            bid_qty=self.bid_qty,
            ask_qty=self.ask_qty,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            mid_history=self.mid_history,
            trade_history=self.trade_history,
            imbalance_history=self.imbalance_history,
            last_trade_ms=self.last_trade_ms or now_ms,
        )
        if snap is None:
            return None

        if self._fh:
            # Written without a label. train.py labels afterwards, so the
            # recorder never needs to know the future and cannot leak it.
            self._fh.write(json.dumps(snap.to_row()) + "\n")
            self._fh.flush()
        self.snapshots_written += 1
        return snap


async def record_live(symbol: str = "BTCUSDT", hours: float = 24.0):
    """Connect to Binance futures and record until the time budget runs out."""
    import websockets

    from config import settings

    sym = symbol.lower()
    # @trade, not @aggTrade. The aggTrade stream returns nothing on this
    # endpoint, which fails silently: depth keeps arriving, trade history stays
    # empty, and build_snapshot returns None forever with no error anywhere.
    streams = f"{sym}@trade/{sym}@depth5@100ms"
    uri = f"{settings.BINANCE_FUTURES_STREAM_URL}?streams={streams}"

    out_path = DATA_DIR / f"{symbol.upper()}_snapshots.jsonl"
    rec = MarketRecorder(symbol=symbol, out_path=out_path)
    rec.open()

    deadline = time.time() + hours * 3600
    logger.info(f"Recording {symbol.upper()} to {out_path} for {hours}h")

    reconnect_delay = 1.0
    n_trades = n_depth = 0
    warned_no_trades = False
    try:
        while time.time() < deadline:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    reconnect_delay = 1.0
                    logger.info("Connected.")
                    while time.time() < deadline:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        payload = data.get("data", data)
                        etype = payload.get("e")

                        if etype in ("trade", "aggTrade"):
                            rec.on_trade(
                                t_ms=int(payload["T"]),
                                price=float(payload["p"]),
                                qty=float(payload["q"]),
                                is_buyer_maker=bool(payload["m"]),
                            )
                            n_trades += 1
                        elif etype == "depthUpdate" or "b" in payload:
                            rec.on_depth(
                                t_ms=int(payload.get("E", time.time() * 1000)),
                                bids=payload.get("b", []),
                                asks=payload.get("a", []),
                            )
                            n_depth += 1

                        # Watchdog. A stream that delivers depth but no trades
                        # produces zero snapshots and no error, so it must be
                        # reported rather than left to run overnight for nothing.
                        if not warned_no_trades and n_depth > 500 and n_trades == 0:
                            warned_no_trades = True
                            logger.error(
                                "Receiving depth but ZERO trades. No snapshots can be built. "
                                "Check the trade stream name for this endpoint."
                            )

                        snap = rec.tick()
                        if snap and rec.snapshots_written % 300 == 0:
                            logger.info(
                                f"{rec.snapshots_written} snapshots | mid={snap.mid:.2f} "
                                f"| trades={n_trades} depth={n_depth} "
                                f"| {(deadline - time.time())/3600:.2f}h left"
                            )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Stream error: {e}. Reconnecting in {reconnect_delay:.0f}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)
    finally:
        rec.close()
        logger.info(f"Done. {rec.snapshots_written} snapshots -> {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Record live market snapshots for training.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    try:
        asyncio.run(record_live(symbol=args.symbol, hours=args.hours))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
