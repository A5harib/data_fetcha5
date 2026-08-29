"""
XGBoost Model Training & Serialization Script.

1. Fetches the last 1000+ 1-minute candles from Binance Futures REST API (with taker buy volume for delta).
2. Computes the feature matrix [1m_CVD_delta, 5m_CVD_delta, orderbook_imbalance, price_change_pct, rsi_14, vwap_distance_pct].
3. Trains a high-precision, lightweight XGBoost binary classifier.
4. Serializes the trained model (.json) and feature scaler (.joblib).
"""
import json
import logging
import sys
from pathlib import Path
import urllib.request

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import settings
from ml.feature_pipeline import generate_feature_matrix_from_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_model")


def fetch_binance_klines(symbol: str = "BTCUSDT", limit: int = 1500) -> pd.DataFrame:
    """Fetch recent historical 1-minute klines from Binance Futures API."""
    url = f"{settings.BINANCE_FUTURES_REST_URL}/fapi/v1/klines?symbol={symbol.upper()}&interval=1m&limit={limit}"
    logger.info(f"Fetching {limit} 1m klines from: {url}")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            
        # Binance Kline fields:
        # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
        # [6] Close time, [7] Quote asset volume, [8] Number of trades,
        # [9] Taker buy base asset volume, [10] Taker buy quote asset volume, [11] Ignore
        records = []
        for k in data:
            vol = float(k[5])
            buy_vol = float(k[9])
            sell_vol = max(0.0, vol - buy_vol)
            records.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": vol,
                "buy_volume": buy_vol,
                "sell_volume": sell_vol
            })
            
        df = pd.DataFrame(records)
        logger.info(f"Successfully loaded {len(df)} klines from Binance.")
        return df
        
    except Exception as e:
        logger.warning(f"Could not fetch live Binance klines ({e}). Generating high-fidelity synthetic market data...")
        return generate_synthetic_data(n_candles=limit)


def generate_synthetic_data(n_candles: int = 1500) -> pd.DataFrame:
    """Generate realistic price series with stochastic CVD and order absorption for training."""
    np.random.seed(42)
    dt = 1.0
    price = 65000.0
    records = []
    
    for i in range(n_candles):
        # Mean-reverting random walk with jump diffusion
        volatility = 0.0015
        ret = np.random.normal(0, volatility)
        open_p = price
        close_p = open_p * (1.0 + ret)
        high_p = max(open_p, close_p) * (1.0 + abs(np.random.normal(0, 0.0005)))
        low_p = min(open_p, close_p) * (1.0 - abs(np.random.normal(0, 0.0005)))
        
        volume = float(np.random.exponential(150.0) + 20.0)
        # Delta correlation with return + occasional divergence
        divergence_event = np.random.rand() < 0.12
        if divergence_event:
            # Opposite delta -> Absorption
            buy_ratio = 0.75 if ret < 0 else 0.25
        else:
            buy_ratio = 0.5 + np.clip(ret * 20.0, -0.4, 0.4)
            
        buy_vol = volume * buy_ratio
        sell_vol = volume - buy_vol
        
        records.append({
            "open_time": 1700000000000 + i * 60000,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol
        })
        price = close_p
        
    return pd.DataFrame(records)


def train_and_save_model(symbol: str = "BTCUSDT"):
    """Pipeline to ingest data, construct features, train XGBoost, and save artifacts."""
    df = fetch_binance_klines(symbol=symbol, limit=1500)
    X, y = generate_feature_matrix_from_df(df, reversal_threshold_pct=0.4)
    
    logger.info(f"Feature matrix shape: {X.shape}, Target distribution: \n{y.value_counts()}")
    
    # Stratified Train-Test Split (Chronological time-series preferred or 80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance weight for rare reversal class
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / max(1, pos_count))
    
    logger.info(f"Training XGBoost Classifier (scale_pos_weight={scale_pos_weight:.2f})...")
    
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
    
    clf.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluation
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    try:
        roc = roc_auc_score(y_test, probs)
        logger.info(f"Test ROC-AUC: {roc:.4f}")
    except Exception as e:
        logger.warning(f"ROC-AUC calculation skipped: {e}")
        
    logger.info("Classification Report:\n" + classification_report(y_test, preds, zero_division=0))
    
    # Feature Importances
    for feat, imp in zip(settings.FEATURE_NAMES, clf.feature_importances_):
        logger.info(f"Feature '{feat}': {imp:.4f}")
        
    # Serialize model & scaler
    clf.save_model(str(settings.MODEL_PATH))
    joblib.dump(scaler, str(settings.SCALER_PATH))
    logger.info(f"Model saved to: {settings.MODEL_PATH}")
    logger.info(f"Scaler saved to: {settings.SCALER_PATH}")


if __name__ == "__main__":
    train_and_save_model()
