"""
Quantitative Analytics & Order Flow Anomaly Engine.

Computes:
1. Bullish & Bearish CVD Divergence / Absorption Signals.
2. Normalized Absorption Score (0 - 100) based on CVD drift vs Price Slope.
3. Fast rolling Technical Features: RSI(14), Session VWAP, Distance %, Price Momentum.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from engine.orderflow_buffer import Candle, OrderFlowBuffer


@dataclass
class DivergenceResult:
    is_bullish_divergence: bool
    is_bearish_divergence: bool
    absorption_score: float  # 0.0 to 100.0
    price_slope: float
    cvd_slope: float
    factors: List[str]


class OrderFlowAnalytics:
    def __init__(self, buffer: OrderFlowBuffer):
        self.buffer = buffer

    def calculate_indicators(self) -> dict:
        """
        Extract key real-time indicators for feature engineering & visualization.
        """
        candles = self.buffer.get_recent_candles(n=30)
        
        # Default fallbacks if insufficient data
        if len(candles) < 2:
            return {
                "rsi_14": 50.0,
                "vwap": self.buffer.last_price or 1.0,
                "vwap_distance_pct": 0.0,
                "price_change_pct": 0.0,
                "cvd_1m_delta": self.buffer.cvd_1m,
                "cvd_5m_delta": self.buffer.cvd_5m,
                "orderbook_imbalance": self.buffer.orderbook_imbalance,
                "orderbook_imbalance_pct": self.buffer.orderbook_imbalance_pct,
            }

        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)

        # 1. 14-period RSI
        rsi = self._compute_rsi(closes, period=14)

        # 2. Rolling VWAP & VWAP Distance %
        typical_price = (highs + lows + closes) / 3.0
        cum_vp = np.sum(typical_price * volumes)
        cum_vol = np.sum(volumes)
        vwap = cum_vp / cum_vol if cum_vol > 0 else closes[-1]
        
        current_price = self.buffer.last_price if self.buffer.last_price > 0 else closes[-1]
        vwap_distance_pct = ((current_price - vwap) / vwap) * 100.0 if vwap > 0 else 0.0

        # 3. 1m Price Change %
        if len(candles) >= 2 and candles[-2].close > 0:
            price_change_pct = ((current_price - candles[-2].close) / candles[-2].close) * 100.0
        else:
            price_change_pct = 0.0

        return {
            "rsi_14": float(rsi),
            "vwap": float(vwap),
            "vwap_distance_pct": float(vwap_distance_pct),
            "price_change_pct": float(price_change_pct),
            "cvd_1m_delta": float(self.buffer.cvd_1m),
            "cvd_5m_delta": float(self.buffer.cvd_5m),
            "orderbook_imbalance": float(self.buffer.orderbook_imbalance),
            "orderbook_imbalance_pct": float(self.buffer.orderbook_imbalance_pct),
        }

    def detect_divergence(self, lookback_candles: int = 5) -> DivergenceResult:
        """
        Detects Order Flow Absorption & Divergence anomalies:
        
        1. Bullish Absorption:
           - Price makes a Lower Low on 1m candles.
           - CVD makes a Higher Low (aggressive market sells absorbed by limit buy walls).
           
        2. Bearish Absorption:
           - Price makes a Higher High on 1m candles.
           - CVD makes a Lower Low / Lower High (aggressive market buys absorbed by limit sell walls).
           
        3. Normalized Absorption Score (0 - 100):
           - Quantifies the divergence degree between normalized price slope and CVD drift.
        """
        candles = self.buffer.get_recent_candles(n=lookback_candles + 1)
        factors: List[str] = []

        if len(candles) < 3:
            return DivergenceResult(
                is_bullish_divergence=False,
                is_bearish_divergence=False,
                absorption_score=0.0,
                price_slope=0.0,
                cvd_slope=0.0,
                factors=factors
            )

        c_prev = candles[-2]
        c_curr = candles[-1]

        # Extract vectors for regression slopes
        prices = np.array([c.close for c in candles], dtype=np.float64)
        cvds = np.array([c.cvd_close for c in candles], dtype=np.float64)
        x = np.arange(len(prices))

        # Linear regression slope (normalized)
        price_std = np.std(prices) if np.std(prices) > 0 else 1.0
        cvd_std = np.std(cvds) if np.std(cvds) > 0 else 1.0

        p_norm = (prices - np.mean(prices)) / price_std
        c_norm = (cvds - np.mean(cvds)) / cvd_std

        price_slope = float(np.polyfit(x, p_norm, 1)[0])
        cvd_slope = float(np.polyfit(x, c_norm, 1)[0])

        # 1. Bullish Divergence check (Lower Low in price, Higher Low in CVD)
        price_lower_low = c_curr.low < c_prev.low or price_slope < -0.2
        cvd_higher_low = c_curr.delta > 0 or cvd_slope > 0.1
        is_bullish_div = bool(price_lower_low and cvd_higher_low)

        # 2. Bearish Divergence check (Higher High in price, Lower Low / Delta in CVD)
        price_higher_high = c_curr.high > c_prev.high or price_slope > 0.2
        cvd_lower_high = c_curr.delta < 0 or cvd_slope < -0.1
        is_bearish_div = bool(price_higher_high and cvd_lower_high)

        # 3. Compute Absorption Score (0 to 100)
        # Absorption occurs when price slope and CVD slope have opposite signs
        slope_diff = cvd_slope - price_slope
        
        if is_bullish_div:
            # Price dropping, CVD rising -> Bullish limit absorption
            raw_score = 50.0 + (slope_diff * 25.0)
            factors.append("CVD Bullish Divergence (Limit Bids Absorbing Market Sells)")
        elif is_bearish_div:
            # Price pushing up, CVD dropping -> Bearish limit absorption
            raw_score = 50.0 + ((-slope_diff) * 25.0)
            factors.append("CVD Bearish Divergence (Limit Asks Absorbing Market Buys)")
        else:
            # Correlated moves or neutral drift
            raw_score = max(0.0, 50.0 - abs(slope_diff) * 15.0)

        # Add Order Book liquidity factors
        if self.buffer.orderbook_imbalance < -0.25:
            factors.append("Order Book Ask Stack (Heavy Resistance Overhead)")
        elif self.buffer.orderbook_imbalance > 0.25:
            factors.append("Order Book Bid Stack (Strong Support Underneath)")

        absorption_score = float(np.clip(raw_score, 0.0, 100.0))

        return DivergenceResult(
            is_bullish_divergence=is_bullish_div,
            is_bearish_divergence=is_bearish_div,
            absorption_score=absorption_score,
            price_slope=price_slope,
            cvd_slope=cvd_slope,
            factors=factors
        )

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Fast Wilder's RSI calculation."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))
