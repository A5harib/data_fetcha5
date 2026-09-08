"""
Configuration and constants for Quantitative Order Flow & ML Backend.
"""
import os
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Retraining writes models back at runtime, so this must point at a writable,
# persistent path in deployment (HF Spaces mounts one at /data). Falls back to
# the in-repo dir for local dev.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
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
    # Legacy single-model paths (BTCUSDT). Kept so existing callers keep working.
    MODEL_PATH: Path = MODEL_DIR / "xgboost_reversal.json"
    SCALER_PATH: Path = MODEL_DIR / "scaler.joblib"

    # Per-symbol artifacts. Volume scale and volatility differ enough between
    # BTC and XAU that one shared scaler puts gold's CVD ~2x outside the fitted
    # range, so each symbol gets its own model + scaler.
    def model_path_for(self, symbol: str) -> Path:
        return MODEL_DIR / f"xgboost_reversal_{symbol.upper()}.json"

    def scaler_path_for(self, symbol: str) -> Path:
        return MODEL_DIR / f"scaler_{symbol.upper()}.joblib"

    # Forward-move threshold defining a "reversal", per symbol, in percent.
    # 0.4% was the old global value; BTC's 99th-percentile 3-bar forward move is
    # ~0.40% and gold's is ~0.20%, so a shared 0.4% labels almost nothing.
    # These are set near each symbol's ~95th percentile instead.
    # Tuned by grid search over (gate, threshold) on purged walk-forward folds.
    # Raising the move a reversal must make, while lowering the gate so far more
    # rows qualify, beat the old pair badly: BTC 0.639 -> 0.764 AUC, XAU
    # 0.661 -> 0.740. The old settings starved the model (638 training rows on
    # BTC fold 0); these give it ~16.6k rows.
    REVERSAL_THRESHOLD_PCT: dict[str, float] = {
        "BTCUSDT": 0.20,
        "XAUUSDT": 0.12,
    }
    DEFAULT_REVERSAL_THRESHOLD_PCT: float = 0.20

    def reversal_threshold_for(self, symbol: str) -> float:
        return self.REVERSAL_THRESHOLD_PCT.get(symbol.upper(), self.DEFAULT_REVERSAL_THRESHOLD_PCT)

    # Minimum prior 3-bar move that makes a bar a reversal candidate. Only these
    # rows are trained and scored on: the rest can never host a reversal, and
    # including them let the model score AUC 0.96 by re-detecting the past move
    # instead of predicting the future. Gold's 3-bar move is ~3x smaller than
    # BTC's, so a shared 0.2% left only 288 usable gold rows out of 36k.
    GATE_PAST_RETURN_PCT: dict[str, float] = {
        "BTCUSDT": 0.05,
        "XAUUSDT": 0.05,
    }
    DEFAULT_GATE_PAST_RETURN_PCT: float = 0.05

    def gate_for(self, symbol: str) -> float:
        return self.GATE_PAST_RETURN_PCT.get(symbol.upper(), self.DEFAULT_GATE_PAST_RETURN_PCT)
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
