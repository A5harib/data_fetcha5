"""
Train and score a short-horizon model from recorded snapshots.

    python -m predict.train --symbol BTCUSDT --horizon 30
    python -m predict.train --symbol BTCUSDT --horizon 30 --kind gbm

The scoreboard runs BEFORE the model is saved, and a model whose walk-forward
IC is not positive is not written to disk. Saving first and evaluating later
is how a useless model ends up serving predictions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from predict.evaluate import evaluate_walk_forward, print_report
from predict.features import (
    FEATURE_NAMES,
    Snapshot,
    label_snapshots,
    snapshots_to_arrays,
)
from predict.model import (
    DEFAULT_HORIZON_SEC,
    FrozenScaler,
    Predictor,
    RidgeCore,
    GBMCore,
    recency_weights,
)

logger = logging.getLogger("predict.train")

DATA_DIR = BASE_DIR / "data"

# Below this, walk-forward folds are too small to mean anything. 2000 seconds
# of recording is roughly half an hour, which is already thin.
MIN_SAMPLES = 2000


def load_snapshots(path: Path) -> List[Snapshot]:
    """Read recorder JSONL back into Snapshot objects."""
    if not path.exists():
        raise FileNotFoundError(
            f"No recording at {path}. Record data first:\n"
            f"  python -m predict.recorder --symbol BTCUSDT --hours 24"
        )

    snaps: List[Snapshot] = []
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                feats = {n: float(row[n]) for n in FEATURE_NAMES}
                snaps.append(Snapshot(t_ms=int(row["t_ms"]), mid=float(row["mid"]), features=feats))
            except Exception:
                bad += 1

    if bad:
        logger.warning(f"Skipped {bad} malformed lines.")
    # Sort by time. Reconnects can interleave, and every window in this
    # pipeline assumes ascending timestamps.
    snaps.sort(key=lambda s: s.t_ms)
    return snaps


def report_data_health(snaps: List[Snapshot]) -> dict:
    """
    Describe the recording before trusting it.

    Gaps matter: a model trained across a two-hour hole has learned from a
    market it never saw. This prints the damage rather than hiding it.
    """
    if len(snaps) < 2:
        return {"n": len(snaps)}

    ts = np.array([s.t_ms for s in snaps], dtype=np.int64)
    gaps = np.diff(ts) / 1000.0
    big = gaps[gaps > 10.0]
    span_h = (ts[-1] - ts[0]) / 3_600_000.0
    coverage = len(snaps) / max(1.0, (ts[-1] - ts[0]) / 1000.0)

    health = {
        "n_snapshots": len(snaps),
        "span_hours": round(span_h, 2),
        "coverage": round(coverage, 3),
        "gaps_over_10s": int(len(big)),
        "largest_gap_sec": round(float(big.max()), 1) if len(big) else 0.0,
    }

    print("Recording health")
    print(f"  snapshots      : {health['n_snapshots']}")
    print(f"  span           : {health['span_hours']}h")
    print(f"  coverage       : {health['coverage']*100:.1f}% of seconds present")
    print(f"  gaps > 10s     : {health['gaps_over_10s']} (largest {health['largest_gap_sec']}s)")
    if coverage < 0.5:
        print("  WARNING: more than half the seconds are missing. Results will be shaky.")
    print()
    return health


def fit_once(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    kind: str = "ridge",
    now_ms: Optional[int] = None,
):
    """Fit scaler and model on one training block. Returns (core, scaler)."""
    scaler = FrozenScaler.fit(X)
    Xs = scaler.transform(X)
    w = recency_weights(timestamps, now_ms=now_ms)
    core = GBMCore() if kind == "gbm" else RidgeCore()
    core.fit(Xs, y, sample_weight=w)
    return core, scaler


def train(
    symbol: str = "BTCUSDT",
    horizon_sec: int = DEFAULT_HORIZON_SEC,
    kind: str = "ridge",
    n_folds: int = 5,
    save: bool = True,
) -> Optional[Predictor]:
    path = DATA_DIR / f"{symbol.upper()}_snapshots.jsonl"
    snaps = load_snapshots(path)
    print(f"Loaded {len(snaps)} snapshots from {path.name}\n")

    report_data_health(snaps)

    snaps = label_snapshots(snaps, horizon_sec=horizon_sec)
    X, y, ts, raw_ret = snapshots_to_arrays(snaps)
    print(f"Labeled {len(y)} usable samples at a {horizon_sec}s horizon "
          f"({len(snaps) - len(y)} unlabeled: tail and gaps)\n")

    if len(y) < MIN_SAMPLES:
        print(f"Not enough data. Need at least {MIN_SAMPLES} labeled samples, have {len(y)}.")
        print("Record more:  python -m predict.recorder --symbol %s --hours 24" % symbol)
        return None

    # Score first. A model that cannot pass the walk-forward does not get saved.
    def fit_predict(X_tr, y_tr, X_te):
        # The fold's own timestamps drive the weights, and "now" is the end of
        # the training block, so a fold is weighted as it would have been live.
        n_tr = len(X_tr)
        core, scaler = fit_once(X_tr, y_tr, ts[:n_tr], kind=kind, now_ms=int(ts[n_tr - 1]))
        return core.predict(scaler.transform(X_te))

    report = evaluate_walk_forward(
        X=X, y=y, timestamps=ts, raw_returns=raw_ret,
        feature_names=FEATURE_NAMES,
        fit_predict_fn=fit_predict,
        n_folds=n_folds,
        horizon_sec=horizon_sec,
    )
    print_report(report, model_name=f"{symbol.upper()} {kind} @ {horizon_sec}s")

    summary = report.summary()
    if not summary:
        print("Evaluation produced no folds. Nothing saved.")
        return None

    # The save gate enforces the same conclusions print_report states. An
    # earlier version only blocked IC <= 0, so a model the report explicitly
    # told you to drop was still written to disk and served.
    baseline_ic = float(np.mean([f.ic_baselines.get("imbalance", 0.0) for f in report.folds]))

    if summary["ic_mean"] <= 0:
        print("Walk-forward IC is not positive. Model not saved.")
        print("This is a real answer, not a failure: these features do not predict "
              f"{horizon_sec}s returns. Try a longer horizon or better features.")
        return None

    if summary["ic_mean"] > 0.15:
        print("Walk-forward IC is implausibly high for this horizon. Model not saved.")
        print("Treat this as a leak until proven otherwise: run predict/test_features.py "
              "and check the recording for gaps.")
        return None

    if summary["ic_mean"] <= baseline_ic:
        print(f"Model IC {summary['ic_mean']:.4f} does not beat book imbalance alone "
              f"({baseline_ic:.4f}). Model not saved.")
        print("Fifteen features and a fitted model lose to reading one number off the "
              "order book. Use the raw imbalance instead.")
        return None

    if summary["ic_positive_folds"] < len(report.folds) * 0.6:
        print(f"Signal appears in only {summary['ic_positive_folds']}/{len(report.folds)} "
              "folds. Too unstable to save.")
        return None

    # Final fit on everything except a recent holdout, which the swap guard uses.
    split = int(len(y) * 0.85)
    core, scaler = fit_once(X[:split], y[:split], ts[:split], kind=kind)

    predictor = Predictor(symbol=symbol)
    predictor.load()  # Load the incumbent, if any, so try_swap can compare.
    accepted = predictor.try_swap(
        candidate_core=core,
        candidate_scaler=scaler,
        X_holdout=X[split:],
        y_holdout=y[split:],
        n_train=split,
        horizon_sec=horizon_sec,
    ) if save else False

    if accepted:
        print(f"Model saved to {predictor.path}")
    elif save:
        print("Candidate did not beat the incumbent on the holdout. Existing model kept.")

    # Feature importance, for ridge, so you can see what it actually leans on.
    if isinstance(core, RidgeCore) and core.coef is not None:
        print("\nStandardized coefficients (largest first):")
        order = np.argsort(-np.abs(core.coef))
        for i in order:
            print(f"  {FEATURE_NAMES[i]:>24} {core.coef[i]:+.4f}")

    return predictor if accepted else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Train a short-horizon predictor from recorded snapshots.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_SEC, help="Seconds ahead to predict.")
    ap.add_argument("--kind", choices=["ridge", "gbm"], default="ridge")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-save", action="store_true", help="Score only, never write a model.")
    args = ap.parse_args()

    train(
        symbol=args.symbol,
        horizon_sec=args.horizon,
        kind=args.kind,
        n_folds=args.folds,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
