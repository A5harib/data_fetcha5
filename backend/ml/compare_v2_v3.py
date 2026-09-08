"""
Before/after comparison: v2 model vs v3, scored on identical walk-forward folds.

The point of this script is that v2's headline AUC of 0.96 and v3's 0.65 are
not comparable -- they answer different questions. v2 was scored on all rows,
96% of which could never be positive, so it earned most of its AUC by
re-detecting the prior 3-bar move. To compare fairly, both models are scored
on the same gated rows (bars where a reversal is actually possible) using the
same purged walk-forward folds.

Run:  python ml/compare_v2_v3.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import settings
from ml.features_v3 import build_dataset, build_frame, HORIZON
from ml.feature_pipeline import generate_feature_matrix_from_df
from ml.train_v3 import load_local_klines, _fit_one, N_FOLDS, EMBARGO

V2_FEATURES = settings.FEATURE_NAMES


def v2_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild v2's six features (including the kline-proxy orderbook imbalance)."""
    d = df.copy()
    d["delta"] = d["buy_volume"] - d["sell_volume"]
    d["cvd_1m_delta"] = d["delta"]
    d["cvd_5m_delta"] = d["delta"].rolling(5, min_periods=1).sum()
    d["orderbook_imbalance"] = np.where(d["volume"] > 0, d["delta"] / d["volume"], 0.0)
    d["price_change_pct"] = d["close"].pct_change() * 100.0
    dp = d["close"].diff()
    gain = dp.where(dp > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-dp.where(dp < 0, 0)).rolling(14, min_periods=1).mean()
    d["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-9))
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    cvp = (tp * d["volume"]).rolling(60, min_periods=1).sum()
    cv = d["volume"].rolling(60, min_periods=1).sum()
    vwap = np.where(cv > 0, cvp / cv, d["close"])
    d["vwap_distance_pct"] = (d["close"] - vwap) / vwap * 100.0
    return d


def aligned_datasets(symbol: str):
    """Return v2 features, v3 features and labels on the SAME gated row index."""
    df = load_local_klines(symbol)
    thr = settings.reversal_threshold_for(symbol)
    gate = settings.gate_for(symbol)

    v3f = build_frame(df)
    v2f = v2_feature_frame(df)

    past3 = df["close"].pct_change(3) * 100.0
    fmax = df["high"].shift(-1).rolling(HORIZON, min_periods=1).max().shift(-(HORIZON - 1))
    fmin = df["low"].shift(-1).rolling(HORIZON, min_periods=1).min().shift(-(HORIZON - 1))
    drop = (fmin - df["close"]) / df["close"] * 100.0
    rise = (fmax - df["close"]) / df["close"] * 100.0
    up, dn = past3 > gate, past3 < -gate
    target = ((up & (drop <= -thr)) | (dn & (rise >= thr))).astype(int)

    from ml.features_v3 import FEATURE_NAMES_V3
    both = pd.concat([v3f[FEATURE_NAMES_V3], v2f[V2_FEATURES].add_prefix("v2_")], axis=1)
    both["target"] = target
    both["_gate"] = up | dn
    both = both.replace([np.inf, -np.inf], np.nan).dropna()
    both = both[both["_gate"]].reset_index(drop=True)

    X3 = both[FEATURE_NAMES_V3]
    X2 = both[[f"v2_{c}" for c in V2_FEATURES]]
    y = both["target"]
    return X2, X3, y


def v2_style_fit(X_tr, y_tr, X_te, y_te):
    """v2's exact recipe: shared scaler, early stopping ON THE TEST SET, no calibration."""
    sc = StandardScaler().fit(X_tr)
    spw = float((y_tr == 0).sum() / max(1, y_tr.sum()))
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                          scale_pos_weight=spw, eval_metric="aucpr",
                          early_stopping_rounds=30, random_state=42)
    m.fit(sc.transform(X_tr), y_tr, eval_set=[(sc.transform(X_te), y_te)], verbose=False)
    return m.predict_proba(sc.transform(X_te))[:, 1]


def compare(symbol: str) -> dict:
    X2, X3, y = aligned_datasets(symbol)
    n = len(X3)
    start = int(n * 0.40)
    fold_size = (n - start) // N_FOLDS

    rows = []
    for i in range(N_FOLDS):
        lo = start + i * fold_size
        hi = lo + fold_size if i < N_FOLDS - 1 else n
        tr_hi = lo - EMBARGO
        if tr_hi < 500 or hi - lo < 50:
            continue
        y_tr, y_te = y.iloc[:tr_hi].values, y.iloc[lo:hi].values
        if y_tr.sum() < 30 or y_te.sum() < 5 or len(np.unique(y_te)) < 2:
            continue

        p2 = v2_style_fit(X2.iloc[:tr_hi].values, y_tr, X2.iloc[lo:hi].values, y_te)
        _, _, p3_raw, p3, _ = _fit_one(X3.iloc[:tr_hi].values, y_tr,
                                       X3.iloc[lo:hi].values, y_te)
        rows.append({
            "fold": i, "base": float(y_te.mean()),
            "v2_auc": roc_auc_score(y_te, p2), "v3_auc": roc_auc_score(y_te, p3),
            "v2_ap": average_precision_score(y_te, p2),
            "v3_ap": average_precision_score(y_te, p3),
            "v2_brier": brier_score_loss(y_te, p2), "v3_brier": brier_score_loss(y_te, p3),
        })

    d = pd.DataFrame(rows)
    return {
        "symbol": symbol, "folds": len(d), "base": d["base"].mean(),
        "v2_auc": d["v2_auc"].mean(), "v3_auc": d["v3_auc"].mean(),
        "v2_auc_std": d["v2_auc"].std(), "v3_auc_std": d["v3_auc"].std(),
        "v2_ap": d["v2_ap"].mean(), "v3_ap": d["v3_ap"].mean(),
        "v2_brier": d["v2_brier"].mean(), "v3_brier": d["v3_brier"].mean(),
        "wins": int((d["v3_auc"] > d["v2_auc"]).sum()),
    }


if __name__ == "__main__":
    syms = sys.argv[1:] or ["BTCUSDT", "XAUUSDT"]
    out = [compare(s) for s in syms]
    print("\n" + "=" * 92)
    print("Same gated rows, same purged walk-forward folds. AUC 0.5 = coin flip.")
    print("=" * 92)
    print(f"{'symbol':<10}{'folds':>6}{'base':>7}{'v2 AUC':>16}{'v3 AUC':>16}"
          f"{'v2 AP':>8}{'v3 AP':>8}{'v3 wins':>9}")
    for r in out:
        a2 = f"{r['v2_auc']:.3f}+-{r['v2_auc_std']:.3f}"
        a3 = f"{r['v3_auc']:.3f}+-{r['v3_auc_std']:.3f}"
        print(f"{r['symbol']:<10}{r['folds']:>6}{r['base']:>7.3f}{a2:>16}{a3:>16}"
              f"{r['v2_ap']:>8.3f}{r['v3_ap']:>8.3f}{r['wins']:>6}/{r['folds']}")
    print()
    for r in out:
        print(f"{r['symbol']}: Brier {r['v2_brier']:.4f} -> {r['v3_brier']:.4f}")
