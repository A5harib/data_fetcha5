"""
Per-symbol XGBoost reversal model training.

Differences from the original train_model.py:
1. Trains one model + scaler per symbol. BTC and XAU have different volume
   scales (ounces vs coins) and different volatility, so a shared StandardScaler
   put ~13% of gold rows outside the fitted range.
2. Reversal threshold is per-symbol (config.REVERSAL_THRESHOLD_PCT). The old
   global 0.4% sat above the 99th percentile of the 3-bar forward move for both
   symbols, so it labeled zero positives and the model learned to always say no.
3. Trains on local paginated history (data/<SYM>_1m.json, ~45k bars) instead of
   a single 1500-bar request.
4. Evaluates against a chronologically held-out tail, and reports precision at
   the 0.70 threshold the predictor actually uses for HIGH_CONVICTION signals.
"""
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from config import settings
from ml.feature_pipeline import generate_feature_matrix_from_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_v2")

DATA_DIR = BASE_DIR / "data"


def load_local_klines(symbol: str) -> pd.DataFrame:
    """Load klines saved by download_data.py."""
    path = DATA_DIR / f"{symbol.upper()}_1m.json"
    if not path.exists():
        raise FileNotFoundError(f"No local data for {symbol}. Run: python download_data.py")

    raw = json.loads(path.read_text())
    records = []
    for k in raw:
        vol = float(k[5])
        buy_vol = float(k[9])
        records.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": vol,
            "buy_volume": buy_vol,
            "sell_volume": max(0.0, vol - buy_vol),
        })

    df = pd.DataFrame(records).drop_duplicates(subset="open_time").sort_values("open_time")
    logger.info(f"{symbol}: loaded {len(df)} bars")
    return df.reset_index(drop=True)


def train_symbol(symbol: str, threshold_pct: float | None = None) -> dict:
    symbol = symbol.upper()
    threshold = threshold_pct if threshold_pct is not None else settings.reversal_threshold_for(symbol)

    df = load_local_klines(symbol)
    X, y = generate_feature_matrix_from_df(df, reversal_threshold_pct=threshold)

    pos = int(y.sum())
    logger.info(f"{symbol}: X={X.shape} threshold={threshold}% positives={pos} ({100*y.mean():.2f}%)")
    if pos < 50:
        logger.warning(f"{symbol}: only {pos} positive labels, model will be weak")

    # Chronological split. Shuffling would leak future bars into training.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    neg, posn = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = float(neg / max(1, posn))

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
    )
    clf.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    probs = clf.predict_proba(X_test_s)[:, 1]
    base_rate = float(y_test.mean())

    result = {"symbol": symbol, "threshold_pct": threshold, "n_rows": len(X),
              "positives": pos, "base_rate_test": base_rate}

    try:
        result["roc_auc"] = float(roc_auc_score(y_test, probs))
    except ValueError:
        result["roc_auc"] = float("nan")

    # Precision at the cutoff the predictor actually uses for HIGH_CONVICTION.
    for cut in (0.50, 0.70):
        pred = (probs >= cut).astype(int)
        n_fired = int(pred.sum())
        result[f"n_signals@{cut}"] = n_fired
        result[f"precision@{cut}"] = float(precision_score(y_test, pred, zero_division=0))
        result[f"recall@{cut}"] = float(recall_score(y_test, pred, zero_division=0))

    logger.info(f"{symbol}: ROC-AUC={result['roc_auc']:.4f} base_rate={base_rate:.4f}")
    logger.info(f"{symbol}: @0.70 fired {result['n_signals@0.7']} times, "
                f"precision={result['precision@0.7']:.4f} recall={result['recall@0.7']:.4f}")
    logger.info("\n" + classification_report(y_test, (probs >= 0.5).astype(int), zero_division=0))

    for feat, imp in zip(settings.FEATURE_NAMES, clf.feature_importances_):
        logger.info(f"  {feat}: {imp:.4f}")

    model_path = settings.model_path_for(symbol)
    scaler_path = settings.scaler_path_for(symbol)
    clf.save_model(str(model_path))
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"{symbol}: saved {model_path.name} + {scaler_path.name}")

    return result


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["BTCUSDT", "XAUUSDT"]
    results = [train_symbol(s) for s in symbols]
    print("\n" + "=" * 78)
    print(f"{'symbol':<10}{'rows':>8}{'pos%':>8}{'AUC':>8}{'sig@.7':>9}{'prec@.7':>10}{'base':>9}")
    print("=" * 78)
    for r in results:
        print(f"{r['symbol']:<10}{r['n_rows']:>8}{100*r['positives']/r['n_rows']:>7.2f}%"
              f"{r['roc_auc']:>8.3f}{r['n_signals@0.7']:>9}"
              f"{r['precision@0.7']:>10.3f}{r['base_rate_test']:>9.3f}")
