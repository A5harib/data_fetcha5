"""
Configuration and constants for Quantitative Order Flow & ML Backend.
"""
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

class Settings(BaseModel):
    # Binance Stream URLs
    BINANCE_FUTURES_WS_URL: str = "wss://fstream.binance.com/ws"
    BINANCE_FUTURES_STREAM_URL: str = "wss://fstream.binance.com/stream"
    BINANCE_FUTURES_REST_URL: str = "https://fapi.binance.com"
    
    # Symbols & Pairs
    DEFAULT_SYMBOL: str = "BTCUSDT"
    
    # Windows in seconds for rolling CVD
    WINDOW_1M_SEC: int = 60
    WINDOW_5M_SEC: int = 300
    WINDOW_15M_SEC: int = 900
    
    # ML Model Configuration
    MODEL_PATH: Path = MODEL_DIR / "xgboost_reversal.json"
    SCALER_PATH: Path = MODEL_DIR / "scaler.joblib"
    FEATURE_NAMES: list[str] = [
        "cvd_1m_delta",
        "cvd_5m_delta",
        "orderbook_imbalance",
        "price_change_pct",
        "rsi_14",
        "vwap_distance_pct"
    ]
    
    # Streaming / Broadcast settings
    BROADCAST_INTERVAL_SEC: float = 0.5  # 500ms real-time updates to frontend

settings = Settings()
