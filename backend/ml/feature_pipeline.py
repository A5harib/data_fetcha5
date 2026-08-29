"""
Feature Engineering & Labeling Pipeline for XGBoost Reversal Model.

Features:
1. cvd_1m_delta: 1-minute Net Volume Delta
2. cvd_5m_delta: 5-minute Net Volume Delta
3. orderbook_imbalance: Normalized Top-5 Bid/Ask Imbalance [-1.0, 1.0]
4. price_change_pct: 1-minute Price Return (%)
5. rsi_14: 14-period Relative Strength Index
6. vwap_distance_pct: Percentage Distance between Current Price and VWAP (%)

Target (y):
- 1 if price reverses by > 0.5% in the subsequent 3 candles (3-minute forward horizon).
- 0 otherwise.
"""
import sys
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd

from config import settings


def extract_features_from_dict(d: dict) -> np.ndarray:
    """Extract a 1D feature array matching configured feature order."""
    return np.array([
        float(d.get("cvd_1m_delta", 0.0)),
        float(d.get("cvd_5m_delta", 0.0)),
        float(d.get("orderbook_imbalance", 0.0)),
        float(d.get("price_change_pct", 0.0)),
        float(d.get("rsi_14", 50.0)),
        float(d.get("vwap_distance_pct", 0.0))
    ], dtype=np.float32)


def generate_feature_matrix_from_df(df: pd.DataFrame, reversal_threshold_pct: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs feature matrix X and target y from historical OHLCV + CVD DataFrame.
    Expected columns: ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
    """
    df = df.copy()
    
    # 1. Delta & CVD
    df['delta'] = df['buy_volume'] - df['sell_volume']
    df['cvd_1m_delta'] = df['delta']
    df['cvd_5m_delta'] = df['delta'].rolling(window=5, min_periods=1).sum()
    
    # 2. Approximate Orderbook Imbalance if not directly recorded
    if 'orderbook_imbalance' not in df.columns:
        # Microstructure proxy: (Buy Vol - Sell Vol) / Total Vol
        df['orderbook_imbalance'] = np.where(
            df['volume'] > 0,
            (df['buy_volume'] - df['sell_volume']) / df['volume'],
            0.0
        )
        
    # 3. 1m Price Change %
    df['price_change_pct'] = df['close'].pct_change() * 100.0
    
    # 4. RSI(14)
    delta_p = df['close'].diff()
    gain = (delta_p.where(delta_p > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta_p.where(delta_p < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 5. Rolling Session VWAP & Distance %
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_vol_price = (typical_price * df['volume']).rolling(window=60, min_periods=1).sum()
    cum_vol = df['volume'].rolling(window=60, min_periods=1).sum()
    df['vwap'] = np.where(cum_vol > 0, cum_vol_price / cum_vol, df['close'])
    df['vwap_distance_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100.0
    
    # -------------------------------------------------------------
    # Target Labeling: Reversal by > 0.5% in next 3 candles
    # -------------------------------------------------------------
    # Forward 3-candle high/low extremes
    forward_max = df['high'].shift(-1).rolling(window=3, min_periods=1).max().shift(-2)
    forward_min = df['low'].shift(-1).rolling(window=3, min_periods=1).min().shift(-2)
    
    # Prior 3-candle trend
    past_return_3m = df['close'].pct_change(3) * 100.0
    
    # Bearish Reversal: Prior uptrend, followed by >= 0.5% drop in next 3 candles
    bearish_rev = (past_return_3m > 0.2) & (((forward_min - df['close']) / df['close'] * 100.0) <= -reversal_threshold_pct)
    
    # Bullish Reversal: Prior downtrend, followed by >= 0.5% bounce in next 3 candles
    bullish_rev = (past_return_3m < -0.2) & (((forward_max - df['close']) / df['close'] * 100.0) >= reversal_threshold_pct)
    
    df['target'] = np.where(bearish_rev | bullish_rev, 1, 0)
    
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = settings.FEATURE_NAMES
    X = df[feature_cols]
    y = df['target']
    
    return X, y
