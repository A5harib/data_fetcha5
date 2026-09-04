"""
Real-time Reversal Predictor & Signal Classifier.

Loads the serialized XGBoost model and produces low-latency inference on live market features.
Maps model probabilities and order flow factors to actionable quant signals:
- HIGH_CONVICTION_BEARISH_REVERSAL
- HIGH_CONVICTION_BULLISH_REVERSAL
- MODERATE_REVERSAL_WATCH
- NEUTRAL_TREND_CONTINUATION
"""
import logging
import sys
from pathlib import Path

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Dict, List, Optional
import numpy as np
import xgboost as xgb
import joblib

from config import settings
from ml.feature_pipeline import extract_features_from_dict

logger = logging.getLogger("predictor")


class ReversalPredictor:
    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None,
                 symbol: Optional[str] = None):
        # A per-symbol model is used when one exists on disk; BTC and XAU have
        # different volume scales, so a shared scaler misplaces gold's CVD.
        # Falls back to the legacy shared artifacts.
        if symbol and not model_path and not scaler_path:
            sym_model = settings.model_path_for(symbol)
            sym_scaler = settings.scaler_path_for(symbol)
            if sym_model.exists() and sym_scaler.exists():
                model_path, scaler_path = sym_model, sym_scaler
            else:
                logger.warning(f"No per-symbol model for {symbol}; using shared model. "
                               f"Train one with: python ml/train_model_v2.py {symbol}")

        self.symbol = symbol
        self.model_path = model_path or settings.MODEL_PATH
        self.scaler_path = scaler_path or settings.SCALER_PATH
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = None
        self._load_or_initialize_model()

    def _load_or_initialize_model(self):
        """Loads serialized model/scaler or bootstraps default model weights."""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(self.model_path))
                self.scaler = joblib.load(str(self.scaler_path))
                logger.info(f"Loaded trained XGBoost model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Error loading model from disk: {e}. Reinitializing default model.")

        # No usable artifacts. Auto-training here used to silently produce a model
        # that predicted "no reversal" for every input, so predict() now returns a
        # neutral 0.5 instead of pretending to have an opinion.
        target = self.symbol or "BTCUSDT"
        logger.error(
            f"No trained model for {target} at {self.model_path}. "
            f"Predictions will be neutral until you run: python ml/train_model_v2.py {target}"
        )

    def predict(self, indicator_dict: dict, active_factors: Optional[List[str]] = None) -> Dict:
        """
        Accepts real-time indicators, standardizes features, and executes inference.
        Returns:
            {
                "probability": float (0.000 to 1.000),
                "signal": str,
                "factors": list[str]
            }
        """
        active_factors = list(active_factors or [])
        features_1d = extract_features_from_dict(indicator_dict).reshape(1, -1)

        # Scale features if scaler exists
        if self.scaler:
            try:
                features_1d = self.scaler.transform(features_1d)
            except Exception:
                pass

        # Inference
        probability = 0.50
        if self.model:
            try:
                prob_arr = self.model.predict_proba(features_1d)[0]
                probability = float(prob_arr[1])
            except Exception as e:
                logger.error(f"Prediction inference error: {e}")
                probability = 0.50

        # Classify reversal direction and signal conviction
        cvd_1m = indicator_dict.get("cvd_1m_delta", 0.0)
        imbalance = indicator_dict.get("orderbook_imbalance", 0.0)
        rsi = indicator_dict.get("rsi_14", 50.0)
        vwap_dist = indicator_dict.get("vwap_distance_pct", 0.0)

        # Identify additional contextual factors
        if rsi > 70.0 and "RSI Overbought (>70)" not in active_factors:
            active_factors.append("RSI Overbought (>70)")
        elif rsi < 30.0 and "RSI Oversold (<30)" not in active_factors:
            active_factors.append("RSI Oversold (<30)")

        if vwap_dist > 0.4 and "Extended VWAP Premium (+0.4%)" not in active_factors:
            active_factors.append("Extended VWAP Premium (+0.4%)")
        elif vwap_dist < -0.4 and "Extended VWAP Discount (-0.4%)" not in active_factors:
            active_factors.append("Extended VWAP Discount (-0.4%)")

        # Determine Signal Label
        is_bearish_divergence = any("Bearish" in f for f in active_factors)
        is_bullish_divergence = any("Bullish" in f for f in active_factors)

        if probability >= 0.70:
            if is_bearish_divergence or (cvd_1m < 0 and imbalance < -0.2) or rsi > 68:
                signal = "HIGH_CONVICTION_BEARISH_REVERSAL"
                if "Order Book Ask Stack" not in active_factors and imbalance < -0.15:
                    active_factors.append("Order Book Ask Stack")
            elif is_bullish_divergence or (cvd_1m > 0 and imbalance > 0.2) or rsi < 32:
                signal = "HIGH_CONVICTION_BULLISH_REVERSAL"
                if "Order Book Bid Stack" not in active_factors and imbalance > 0.15:
                    active_factors.append("Order Book Bid Stack")
            else:
                signal = "HIGH_CONVICTION_REVERSAL_ALERT"
        elif probability >= 0.55:
            signal = "MODERATE_REVERSAL_WATCH"
        else:
            signal = "NEUTRAL_TREND_CONTINUATION"

        return {
            "probability": round(probability, 3),
            "signal": signal,
            "factors": active_factors
        }
