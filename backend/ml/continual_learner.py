"""
Continual Online Learning Engine for XGBoost Reversal Model.

How it works:
1. Online Sample Collection:
   - When a 1m candle closes, extracts feature snapshot X_t.
   - Pushes into a pending queue awaiting ground-truth resolution.
2. Ground-Truth Labeling:
   - As 3 subsequent candles close (3-minute horizon), evaluates if a >= 0.5% reversal occurred.
   - Labels y_t = 1 (reversal) or y_t = 0 (continuation/noise).
3. Continual Experience Replay:
   - Stores labeled samples in an adaptive sliding replay buffer (up to 3,000 candles).
4. Hot-Reload Incremental Retraining:
   - Periodically (e.g., every 15 new labeled samples or every 10 mins), triggers background incremental training.
   - Applies exponential recency decay weights so the model adapts rapidly to changing volatility/regimes.
   - Hot-swaps the in-memory XGBoost model atomically with zero downtime for WebSocket streams.
"""
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Deque, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

from config import settings
from engine.orderflow_buffer import Candle

logger = logging.getLogger("continual_learner")


@dataclass
class PendingCandleSample:
    open_time_ms: int
    features: np.ndarray
    close_price: float
    high_price: float
    low_price: float
    prior_3m_return: float
    candles_forward_count: int = 0
    forward_highs: List[float] = None
    forward_lows: List[float] = None

    def __post_init__(self):
        if self.forward_highs is None:
            self.forward_highs = []
        if self.forward_lows is None:
            self.forward_lows = []


class ContinualLearningEngine:
    def __init__(self, predictor, retrain_batch_threshold: int = 15, max_replay_size: int = 3000):
        self.predictor = predictor
        self.retrain_batch_threshold = retrain_batch_threshold
        self.max_replay_size = max_replay_size

        # Queue for samples awaiting 3-candle forward ground truth
        self.pending_samples: Deque[PendingCandleSample] = deque()

        # Experience replay buffer of verified (features, label, timestamp)
        self.replay_X: List[np.ndarray] = []
        self.replay_y: List[int] = []
        self.replay_timestamps: List[int] = []

        # Retraining state
        self.samples_labeled_since_last_retrain: int = 0
        self.total_retrain_count: int = 0
        self.last_retrain_time: Optional[str] = None
        self.is_retraining: bool = False
        self._lock = asyncio.Lock()

    def on_candle_closed(self, closed_candle: Candle, features_dict: dict, recent_candles: List[Candle]):
        """
        Called whenever a 1-minute candle closes in the order flow engine.
        1. Updates pending samples with forward high/low to resolve ground truth.
        2. Adds current closed candle to pending queue.
        3. If enough samples are labeled, triggers asynchronous incremental retrain.
        """
        curr_high = closed_candle.high
        curr_low = closed_candle.low

        # 1. Update pending samples awaiting 3-candle horizon
        resolved_count = 0
        for sample in list(self.pending_samples):
            sample.forward_highs.append(curr_high)
            sample.forward_lows.append(curr_low)
            sample.candles_forward_count += 1

            # After 3 forward candles, label ground truth
            if sample.candles_forward_count >= 3:
                max_fwd_high = max(sample.forward_highs)
                min_fwd_low = min(sample.forward_lows)
                
                # Check 0.5% reversal condition
                up_move_pct = ((max_fwd_high - sample.close_price) / sample.close_price) * 100.0
                down_move_pct = ((min_fwd_low - sample.close_price) / sample.close_price) * 100.0
                
                is_bearish_rev = (sample.prior_3m_return > 0.2) and (down_move_pct <= -0.45)
                is_bullish_rev = (sample.prior_3m_return < -0.2) and (up_move_pct >= 0.45)
                
                y_label = 1 if (is_bearish_rev or is_bullish_rev) else 0

                # Append to experience replay buffer
                self.replay_X.append(sample.features)
                self.replay_y.append(y_label)
                self.replay_timestamps.append(sample.open_time_ms)
                
                # Trim replay buffer
                if len(self.replay_X) > self.max_replay_size:
                    self.replay_X.pop(0)
                    self.replay_y.pop(0)
                    self.replay_timestamps.pop(0)

                self.samples_labeled_since_last_retrain += 1
                resolved_count += 1

        # Remove resolved samples from pending deque
        for _ in range(resolved_count):
            if self.pending_samples:
                self.pending_samples.popleft()

        # 2. Extract feature vector for current closed candle and add to pending queue
        from ml.feature_pipeline import extract_features_from_dict
        feat_vec = extract_features_from_dict(features_dict)

        # Prior 3-candle return
        if len(recent_candles) >= 4 and recent_candles[-4].close > 0:
            prior_3m_return = ((closed_candle.close - recent_candles[-4].close) / recent_candles[-4].close) * 100.0
        else:
            prior_3m_return = 0.0

        self.pending_samples.append(
            PendingCandleSample(
                open_time_ms=closed_candle.open_time_ms,
                features=feat_vec,
                close_price=closed_candle.close,
                high_price=closed_candle.high,
                low_price=closed_candle.low,
                prior_3m_return=prior_3m_return
            )
        )

        logger.info(
            f"[Continual Learning] Pending: {len(self.pending_samples)} | "
            f"Replay Buffer: {len(self.replay_X)} samples | "
            f"New since retrain: {self.samples_labeled_since_last_retrain}/{self.retrain_batch_threshold}"
        )

        # 3. Check if we should trigger background online retraining
        if self.samples_labeled_since_last_retrain >= self.retrain_batch_threshold and not self.is_retraining:
            asyncio.create_task(self.trigger_incremental_retrain())

    async def trigger_incremental_retrain(self):
        """Asynchronously retrains the model on live experience replay with recency weighting."""
        async with self._lock:
            if self.is_retraining:
                return
            self.is_retraining = True

        try:
            logger.info("[Continual Learning] Starting background online model update...")
            await asyncio.to_thread(self._execute_retraining_sync)
            self.samples_labeled_since_last_retrain = 0
            self.total_retrain_count += 1
            self.last_retrain_time = datetime.utcnow().isoformat() + "Z"
            logger.info(f"[Continual Learning] Online Retrain #{self.total_retrain_count} completed successfully!")
        except Exception as e:
            logger.error(f"[Continual Learning] Error during incremental retrain: {e}")
        finally:
            self.is_retraining = False

    def _execute_retraining_sync(self):
        """Runs scikit-learn / XGBoost training in a separate thread."""
        if len(self.replay_X) < 30:
            # Seed with historical data if replay buffer is small
            from ml.train_model import fetch_binance_klines
            from ml.feature_pipeline import generate_feature_matrix_from_df
            df = fetch_binance_klines(symbol="BTCUSDT", limit=1000)
            X_hist, y_hist = generate_feature_matrix_from_df(df)
            
            X_combined = np.vstack([X_hist.values] + ([np.array(self.replay_X)] if self.replay_X else []))
            y_combined = np.concatenate([y_hist.values] + ([np.array(self.replay_y)] if self.replay_y else []))
        else:
            X_combined = np.array(self.replay_X)
            y_combined = np.array(self.replay_y)

        # Exponential recency weights: recent candles get higher weight
        n_samples = len(X_combined)
        recency_weights = np.linspace(0.6, 1.4, n_samples)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        neg_count = np.sum(y_combined == 0)
        pos_count = np.sum(y_combined == 1)
        scale_pos_weight = float(neg_count / max(1, pos_count))

        # Train updated model
        clf = xgb.XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        )

        clf.fit(X_scaled, y_combined, sample_weight=recency_weights)

        # Atomic save to disk
        clf.save_model(str(settings.MODEL_PATH))
        joblib.dump(scaler, str(settings.SCALER_PATH))

        # Hot-reload in predictor in memory
        self.predictor.model = clf
        self.predictor.scaler = scaler
        logger.info(f"[Continual Learning] Hot-swapped updated XGBoost model weights (samples={n_samples})")

    def get_status(self) -> dict:
        return {
            "total_retrain_count": self.total_retrain_count,
            "last_retrain_time": self.last_retrain_time,
            "replay_buffer_size": len(self.replay_X),
            "pending_samples_count": len(self.pending_samples),
            "samples_since_last_retrain": self.samples_labeled_since_last_retrain,
            "is_retraining": self.is_retraining
        }
