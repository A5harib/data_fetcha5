"""
Binance WebSocket Stream Ingestion Client.

Subscribes to:
1. <symbol>@aggTrade (Aggregated Real-time Ticks)
2. <symbol>@depth5@100ms or <symbol>@depth5 (Order Book Top 5 Level 2 Depth)

Supports Binance Futures (fapi / fstream) with auto-reconnection and error resilience.
"""
import asyncio
import json
import logging
from typing import Callable, Optional
import websockets
from websockets.exceptions import ConnectionClosed

from config import settings
from engine.orderflow_buffer import OrderFlowBuffer

logger = logging.getLogger("binance_stream")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class BinanceFuturesStreamConsumer:
    def __init__(self, symbol: str, buffer: OrderFlowBuffer):
        self.symbol: str = symbol.lower()
        self.buffer: OrderFlowBuffer = buffer
        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._on_tick_callbacks: list[Callable[[], None]] = []

    def register_on_tick_callback(self, callback: Callable[[], None]):
        """Register a callback executed upon new market data updates."""
        self._on_tick_callbacks.append(callback)

    async def start(self):
        """Start streaming Binance data as an asynchronous background task."""
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info(f"Binance consumer started for {self.symbol.upper()}")

    async def stop(self):
        """Stop streaming and cleanup tasks."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Binance consumer stopped for {self.symbol.upper()}")

    async def _stream_loop(self):
        """
        Combined stream endpoint on Binance Futures:
        Streams: <symbol>@aggTrade / <symbol>@depth5@100ms
        """
        # @trade, not @aggTrade: the aggTrade stream returns no messages on
        # this endpoint, leaving CVD permanently at zero with no error raised.
        stream_names = f"{self.symbol}@trade/{self.symbol}@depth5@100ms"
        uri = f"{settings.BINANCE_FUTURES_STREAM_URL}?streams={stream_names}"

        reconnect_delay = 1.0
        while self.is_running:
            try:
                logger.info(f"Connecting to Binance Futures WS: {uri}")
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**22
                ) as ws:
                    reconnect_delay = 1.0  # Reset delay on successful connection
                    logger.info(f"Connected to Binance Futures Stream for {self.symbol.upper()}")

                    while self.is_running:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            self._process_message(data)
                            
                            # Fire registered callbacks (if any)
                            for cb in self._on_tick_callbacks:
                                try:
                                    cb()
                                except Exception as cb_err:
                                    logger.error(f"Callback error: {cb_err}")
                        except ConnectionClosed as e:
                            logger.warning(f"WS connection closed: {e}. Reconnecting...")
                            break
                        except Exception as e:
                            logger.error(f"Error processing packet: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WS connection failed: {e}. Retrying in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    def _process_message(self, data: dict):
        """Route message to appropriate stream handler."""
        stream = data.get("stream", "")
        payload = data.get("data", {})

        if not payload:
            payload = data

        event_type = payload.get("e")
        
        # 1. Handle Aggregated Trade (@aggTrade)
        if event_type in ("trade", "aggTrade") or "@trade" in stream:
            price = float(payload["p"])
            quantity = float(payload["q"])
            is_buyer_maker = bool(payload["m"])
            trade_time = int(payload["T"])
            
            self.buffer.add_agg_trade(
                price=price,
                quantity=quantity,
                is_buyer_maker=is_buyer_maker,
                trade_time_ms=trade_time
            )

        # 2. Handle Order Book Depth Top 5 (@depth5)
        elif event_type == "depthUpdate" or "@depth" in stream:
            bids = payload.get("b", []) or payload.get("bids", [])
            asks = payload.get("a", []) or payload.get("asks", [])
            if bids and asks:
                self.buffer.update_depth(bids=bids, asks=asks)
