"""
Walk-forward training and honest evaluation for the reversal model.

What changed from ml/train_model_v2.py, and why:

1. Purged, embargoed walk-forward instead of one 80/20 cut. Positives cluster
   in time (BTC's 6th time-decile held 5.6% positives against 0.1-0.2% in the
   first five), so a single split reports the weather on one week rather than
   the model. Each fold trains on everything before the test window, minus an
   embargo of HORIZON bars so no training label peeks into the test window.

2. Early stopping uses a validation slice carved from the *tail of training*,
   never the test fold. v2 passed the test set as `eval_set`, which let the
   held-out data choose the tree count.

3. Probabilities are calibrated with isotonic regression fitted on that same
   validation slice. `scale_pos_weight` deliberately distorts probabilities,
   so the raw 0.70 cutoff in the predictor never meant 70%.

4. The decision threshold is chosen from the validation precision-recall
   curve at a target precision, not hardcoded to a round number.

Run:  python ml/train_v3.py BTCUSDT XAUUSDT
"""
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import settings, MODEL_DIR
from ml.features_v3 import FEATURE_NAMES_V3, build_dataset, HORIZON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v3")

DATA_DIR = BASE_DIR / "data"
N_FOLDS = 6
EMBARGO = HORIZON          # bars dropped between train and test
VAL_FRAC = 0.15            # tail of train used for early stopping + calibration
TARGET_PRECISION = 0.45    # threshold picked at this precision on validation

PARAMS = dict(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=20,
    reg_lambda=3.0,
    reg_alpha=0.5,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
)


def load_local_klines(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.upper()}_1m.json"
    if not path.exists():
        raise FileNotFoundError(f"No local data for {symbol}. Run: python download_data.py")
    raw = json.loads(path.read_text())
    rec = []
    for k in raw:
        vol, buy = float(k[5]), float(k[9])
        rec.append({"open_time": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": vol,
                    "buy_volume": buy, "sell_volume": max(0.0, vol - buy)})
    df = pd.DataFrame(rec).drop_duplicates(subset="open_time").sort_values("open_time")
    return df.reset_index(drop=True)


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    ap: float
    lift: float          # average_precision / base_rate
    brier_raw: float
    brier_cal: float
    threshold: float
    n_fired: int
    precision: float
    recall: float


def _pick_threshold(y_val, p_val, target=TARGET_PRECISION):
    """Lowest threshold whose validation precision clears `target`, else the best available."""
    best_t, best_p = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        fired = p_val >= t
        if fired.sum() < 10:
            continue
        prec = float(y_val[fired].mean())
        if prec >= target:
            return float(t)
        if prec > best_p:
            best_p, best_t = prec, float(t)
    return best_t


def _fit_one(X_tr, y_tr, X_te, y_te):
    """Fit with an internal validation tail; return calibrated test probs + pieces."""
    n_val = max(200, int(len(X_tr) * VAL_FRAC))
    n_val = min(n_val, len(X_tr) // 3)
    X_fit, y_fit = X_tr[:-n_val], y_tr[:-n_val]
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]

    pos = max(1, int(y_fit.sum()))
    spw = float((y_fit == 0).sum() / pos)

    clf = xgb.XGBClassifier(**PARAMS, scale_pos_weight=spw,
                            early_stopping_rounds=50)
    clf.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_te_raw = clf.predict_proba(X_te)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, y_val)
    p_val = iso.predict(p_val_raw)
    p_te = iso.predict(p_te_raw)

    thr = _pick_threshold(np.asarray(y_val), p_val)
    return clf, iso, p_te_raw, p_te, thr


def walk_forward(symbol: str, gated: bool = True) -> list[FoldResult]:
    df = load_local_klines(symbol)
    X, y, _ = build_dataset(df, settings.reversal_threshold_for(symbol), gated=gated,
                            gate_pct=settings.gate_for(symbol))
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    n = len(X)
    logger.info(f"{symbol}: {n} rows, base rate {y.mean():.4f}, {int(y.sum())} positives")

    results = []
    # Expanding window: first fold trains on 40% of history, each later fold adds one slice.
    start = int(n * 0.40)
    fold_size = (n - start) // N_FOLDS
    for i in range(N_FOLDS):
        te_lo = start + i * fold_size
        te_hi = te_lo + fold_size if i < N_FOLDS - 1 else n
        tr_hi = te_lo - EMBARGO
        if tr_hi < 500 or te_hi - te_lo < 50:
            continue

        X_tr, y_tr = X.iloc[:tr_hi].values, y.iloc[:tr_hi].values
        X_te, y_te = X.iloc[te_lo:te_hi].values, y.iloc[te_lo:te_hi].values
        if y_tr.sum() < 30 or y_te.sum() < 5:
            continue

        clf, iso, p_raw, p_cal, thr = _fit_one(X_tr, y_tr, X_te, y_te)

        base = float(y_te.mean())
        fired = p_cal >= thr
        results.append(FoldResult(
            fold=i, n_train=len(X_tr), n_test=len(X_te), base_rate=base,
            auc=float(roc_auc_score(y_te, p_cal)) if len(np.unique(y_te)) > 1 else float("nan"),
            ap=float(average_precision_score(y_te, p_cal)),
            lift=float(average_precision_score(y_te, p_cal) / max(base, 1e-9)),
            brier_raw=float(brier_score_loss(y_te, p_raw)),
            brier_cal=float(brier_score_loss(y_te, p_cal)),
            threshold=thr, n_fired=int(fired.sum()),
            precision=float(y_te[fired].mean()) if fired.sum() else float("nan"),
            recall=float(y_te[fired].sum() / max(1, y_te.sum())),
        ))
    return results


def summarize(symbol: str, res: list[FoldResult]) -> dict:
    if not res:
        return {"symbol": symbol, "folds": 0}
    g = lambda k: np.array([getattr(r, k) for r in res], dtype=float)
    out = {
        "symbol": symbol, "folds": len(res),
        "base_rate": float(np.nanmean(g("base_rate"))),
        "auc_mean": float(np.nanmean(g("auc"))), "auc_std": float(np.nanstd(g("auc"))),
        "auc_min": float(np.nanmin(g("auc"))), "auc_max": float(np.nanmax(g("auc"))),
        "ap_mean": float(np.nanmean(g("ap"))),
        "lift_mean": float(np.nanmean(g("lift"))),
        "brier_raw": float(np.nanmean(g("brier_raw"))),
        "brier_cal": float(np.nanmean(g("brier_cal"))),
        "precision_mean": float(np.nanmean(g("precision"))),
        "recall_mean": float(np.nanmean(g("recall"))),
        "fired_total": int(np.nansum(g("n_fired"))),
    }
    return out


def fit_final(symbol: str):
    """Train the deployed artifacts on all history, calibrated on the tail."""
    df = load_local_klines(symbol)
    X, y, _ = build_dataset(df, settings.reversal_threshold_for(symbol), gated=True,
                            gate_pct=settings.gate_for(symbol))
    X, y = X.values, y.values

    clf, iso, _, _, thr = _fit_one(X, y, X[-10:], y[-10:])

    mp = MODEL_DIR / f"xgb_v3_{symbol.upper()}.json"
    cp = MODEL_DIR / f"calib_v3_{symbol.upper()}.joblib"
    clf.save_model(str(mp))
    joblib.dump({"isotonic": iso, "threshold": thr, "features": FEATURE_NAMES_V3}, str(cp))
    logger.info(f"{symbol}: saved {mp.name} + {cp.name} (threshold {thr:.3f})")
    return mp, cp, thr


if __name__ == "__main__":
    syms = [s for s in sys.argv[1:] if not s.startswith("-")] or ["BTCUSDT", "XAUUSDT"]
    rows = []
    for s in syms:
        logger.info(f"===== {s} =====")
        res = walk_forward(s, gated=True)
        for r in res:
            logger.info(f"  fold {r.fold}: base={r.base_rate:.3f} AUC={r.auc:.3f} "
                        f"AP={r.ap:.3f} thr={r.threshold:.2f} fired={r.n_fired} "
                        f"prec={r.precision:.3f} rec={r.recall:.3f}")
        summ = summarize(s, res)
        rows.append(summ)
        fit_final(s)

    print("\n" + "=" * 96)
    print(f"{'symbol':<10}{'folds':>6}{'base':>8}{'AUC':>18}{'AP':>8}{'lift':>7}"
          f"{'prec':>8}{'rec':>7}{'brier':>16}")
    print("=" * 96)
    for r in rows:
        if not r.get("folds"):
            print(f"{r['symbol']:<10}  no usable folds")
            continue
        auc = f"{r['auc_mean']:.3f}+-{r['auc_std']:.3f}"
        brier = f"{r['brier_raw']:.3f}->{r['brier_cal']:.3f}"
        print(f"{r['symbol']:<10}{r['folds']:>6}{r['base_rate']:>8.3f}{auc:>18}"
              f"{r['ap_mean']:>8.3f}{r['lift_mean']:>7.2f}"
              f"{r['precision_mean']:>8.3f}{r['recall_mean']:>7.3f}{brier:>16}")
    (BASE_DIR / "data" / "v3_metrics.json").write_text(json.dumps(rows, indent=2))
