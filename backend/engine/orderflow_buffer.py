"""
Order Flow State & Rolling CVD Buffer.

Maintains high-performance in-memory ring buffers for:
- Raw tick deltas (1m, 5m, 15m sliding windows)
- Completed and running 1-minute OHLCV + CVD candles
- Level 2 Depth (top 5 bids/asks) and order book liquidity imbalance ratio
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TradeTick:
    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool  # True: Sell market order, False: Buy market order
    delta: float          # +quantity for Buy, -quantity for Sell


@dataclass
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    delta: float        # Net delta (buy_vol - sell_vol)
    cvd_close: float    # Cumulative Volume Delta at candle close
    is_closed: bool = False


class OrderFlowBuffer:
    def __init__(self, symbol: str):
        self.symbol: str = symbol.upper()
        
        # Latest market price
        self.last_price: float = 0.0
        self.last_trade_time_ms: int = 0
        
        # Continuous lifetime CVD tracker
        self.lifetime_cvd: float = 0.0
        
        # Sliding tick buffers for precise delta windows (stores TradeTick)
        self._ticks_15m: deque[TradeTick] = deque()
        
        # Fast rolling sums for O(1) CVD delta retrieval
        self._cvd_sum_1m: float = 0.0
        self._cvd_sum_5m: float = 0.0
        self._cvd_sum_15m: float = 0.0
        
        # Order book depth state (top 5 bids/asks)
        self.top_bids: List[Tuple[float, float]] = []  # [(price, qty), ...]
        self.top_asks: List[Tuple[float, float]] = []  # [(price, qty), ...]
        self.orderbook_imbalance: float = 0.0          # Range [-1.0, 1.0]
        self.orderbook_imbalance_pct: float = 0.0      # Range [-100.0, 100.0]
        
        # Candle historical buffer (last 1200 1-minute candles for feature calc & divergence)
        self.candles_1m: deque[Candle] = deque(maxlen=1200)
        self.current_candle: Optional[Candle] = None

    def add_agg_trade(self, price: float, quantity: float, is_buyer_maker: bool, trade_time_ms: int) -> TradeTick:
        """
        Process an aggregated trade from Binance stream.
        Microstructure logic:
          - If is_buyer_maker is True: Taker is SELLER -> Market Sell -> delta = -quantity
          - If is_buyer_maker is False: Taker is BUYER -> Market Buy -> delta = +quantity
        """
        delta = -quantity if is_buyer_maker else quantity
        tick = TradeTick(
            timestamp_ms=trade_time_ms,
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            delta=delta
        )
        
        self.last_price = price
        self.last_trade_time_ms = trade_time_ms
        self.lifetime_cvd += delta
        
        # Append to 15m tick deque
        self._ticks_15m.append(tick)
        
        # Update 1m candle formation
        self._update_candle(tick)
        
        # Prune buffers and maintain window sums
        self._prune_expired_ticks(trade_time_ms)
        
        return tick

    def _update_candle(self, tick: TradeTick):
        """Update or roll over 1-minute OHLCV + CVD candle."""
        candle_open_ms = (tick.timestamp_ms // 60000) * 60000
        candle_close_ms = candle_open_ms + 60000 - 1

        if self.current_candle is None:
            self.current_candle = Candle(
                open_time_ms=candle_open_ms,
                close_time_ms=candle_close_ms,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.quantity,
                buy_volume=tick.quantity if not tick.is_buyer_maker else 0.0,
                sell_volume=tick.quantity if tick.is_buyer_maker else 0.0,
                delta=tick.delta,
                cvd_close=self.lifetime_cvd,
                is_closed=False
            )
        elif self.current_candle.open_time_ms == candle_open_ms:
            # Update current running candle
            c = self.current_candle
            c.high = max(c.high, tick.price)
            c.low = min(c.low, tick.price)
            c.close = tick.price
            c.volume += tick.quantity
            if not tick.is_buyer_maker:
                c.buy_volume += tick.quantity
            else:
                c.sell_volume += tick.quantity
            c.delta += tick.delta
            c.cvd_close = self.lifetime_cvd
        else:
            # Candle period completed -> finalize previous candle and start new one
            self.current_candle.is_closed = True
            self.candles_1m.append(self.current_candle)
            
            # Start fresh candle
            self.current_candle = Candle(
                open_time_ms=candle_open_ms,
                close_time_ms=candle_close_ms,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.quantity,
                buy_volume=tick.quantity if not tick.is_buyer_maker else 0.0,
                sell_volume=tick.quantity if tick.is_buyer_maker else 0.0,
                delta=tick.delta,
                cvd_close=self.lifetime_cvd,
                is_closed=False
            )

    def _prune_expired_ticks(self, current_time_ms: int):
        """Prune ticks older than 15 minutes and calculate rolling CVD sums."""
        cutoff_1m = current_time_ms - 60_000
        cutoff_5m = current_time_ms - 300_000
        cutoff_15m = current_time_ms - 900_000
        
        # Evict ticks older than 15 minutes
        while self._ticks_15m and self._ticks_15m[0].timestamp_ms < cutoff_15m:
            self._ticks_15m.popleft()
            
        # Re-compute window sums efficiently
        sum_1m = 0.0
        sum_5m = 0.0
        sum_15m = 0.0
        
        for t in reversed(self._ticks_15m):
            sum_15m += t.delta
            if t.timestamp_ms >= cutoff_5m:
                sum_5m += t.delta
            if t.timestamp_ms >= cutoff_1m:
                sum_1m += t.delta
                
        self._cvd_sum_1m = sum_1m
        self._cvd_sum_5m = sum_5m
        self._cvd_sum_15m = sum_15m

    def update_depth(self, bids: List[List[str]], asks: List[List[str]]):
        """
        Update Top-5 Bid/Ask Order Book Depth and Liquidity Imbalance.
        
        Liquidity Imbalance Formula:
            Imbalance = (Total Bid Vol - Total Ask Vol) / (Total Bid Vol + Total Ask Vol)
            Normalized between -1.0 (-100%) [Extreme Ask Wall] and +1.0 (+100%) [Extreme Bid Wall]
        """
        self.top_bids = [(float(p), float(q)) for p, q in bids[:5]]
        self.top_asks = [(float(p), float(q)) for p, q in asks[:5]]
        
        total_bid_vol = sum(q for _, q in self.top_bids)
        total_ask_vol = sum(q for _, q in self.top_asks)
        total_vol = total_bid_vol + total_ask_vol
        
        if self.last_price == 0.0 and self.top_bids and self.top_asks:
            self.last_price = (self.top_bids[0][0] + self.top_asks[0][0]) / 2.0
        
        if total_vol > 0:
            self.orderbook_imbalance = (total_bid_vol - total_ask_vol) / total_vol
            self.orderbook_imbalance_pct = self.orderbook_imbalance * 100.0
        else:
            self.orderbook_imbalance = 0.0
            self.orderbook_imbalance_pct = 0.0

    @property
    def cvd_1m(self) -> float:
        return self._cvd_sum_1m

    @property
    def cvd_5m(self) -> float:
        return self._cvd_sum_5m

    @property
    def cvd_15m(self) -> float:
        return self._cvd_sum_15m

    def get_recent_candles(self, n: int = 100) -> List[Candle]:
        """Returns a list of completed candles + current candle up to n."""
        candles = list(self.candles_1m)
        if self.current_candle:
            candles.append(self.current_candle)
        return candles[-n:]
