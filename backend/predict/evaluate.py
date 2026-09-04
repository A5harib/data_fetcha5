"""
Honest scoring for a short-horizon prediction model.

Two rules this file exists to enforce:

1. A model is only as good as the dumb thing it beats. Every metric is
   reported next to three baselines. Beating none of them means the model
   is decoration.
2. Fees come out before anyone celebrates. At a 30s horizon the edge is a
   few basis points and taker fees are 4bp a side, so a "profitable" model
   measured without costs is fiction.

Expect small numbers. Information coefficient of 0.02-0.06 on 30s crypto
returns is a real result. Anything above 0.15 is a leak until proven
otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Binance USD-M futures taker fee, one side. Round trip is twice this.
TAKER_FEE_BPS = 4.0
# Crossing the spread on a liquid pair. Deliberately pessimistic.
SLIPPAGE_BPS = 1.0


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    ic: float                 # correlation, prediction vs realized
    ic_baselines: Dict[str, float]
    top_decile_hit: float     # direction accuracy on the most confident 10%
    base_rate: float          # accuracy of always guessing the majority direction
    net_bps_per_trade: float  # after fees, on the traded subset
    n_trades: int


@dataclass
class EvalReport:
    folds: List[FoldResult] = field(default_factory=list)
    horizon_sec: int = 30
    threshold_pct: float = 90.0

    def summary(self) -> dict:
        if not self.folds:
            return {}
        ics = np.array([f.ic for f in self.folds])
        hits = np.array([f.top_decile_hit for f in self.folds])
        base = np.array([f.base_rate for f in self.folds])
        net = np.array([f.net_bps_per_trade for f in self.folds])
        return {
            "n_folds": len(self.folds),
            "ic_mean": float(np.mean(ics)),
            "ic_std": float(np.std(ics)),
            "ic_positive_folds": int(np.sum(ics > 0)),
            "top_decile_hit_mean": float(np.mean(hits)),
            "base_rate_mean": float(np.mean(base)),
            "edge_over_base": float(np.mean(hits - base)),
            "net_bps_mean": float(np.mean(net)),
            "total_trades": int(sum(f.n_trades for f in self.folds)),
        }


def information_coefficient(pred: np.ndarray, actual: np.ndarray) -> float:
    """Correlation between prediction and outcome. The headline number."""
    if len(pred) < 10:
        return 0.0
    if np.std(pred) < 1e-12 or np.std(actual) < 1e-12:
        return 0.0
    c = float(np.corrcoef(pred, actual)[0, 1])
    return c if np.isfinite(c) else 0.0


def baseline_predictions(X: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Three things a model must beat to justify existing.

    zero:      predict no move, ever. Scores 0 IC by construction, and its
               direction accuracy is the base rate the model must clear.
    momentum:  assume the last 30s return continues.
    imbalance: assume the order book lean predicts the next move. This is
               the strongest of the three and the real bar.
    """
    n = len(X)
    out = {"zero": np.zeros(n, dtype=np.float64)}
    idx = {name: i for i, name in enumerate(feature_names)}
    out["momentum"] = X[:, idx["ret_30s"]].astype(np.float64) if "ret_30s" in idx else np.zeros(n)
    out["imbalance"] = X[:, idx["book_imbalance"]].astype(np.float64) if "book_imbalance" in idx else np.zeros(n)
    return out


def direction_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of non-flat predictions that got the sign right."""
    mask = (np.abs(pred) > 1e-12) & (np.abs(actual) > 1e-12)
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(np.sign(pred[mask]) == np.sign(actual[mask])))


def majority_base_rate(actual: np.ndarray) -> float:
    """
    Accuracy of always betting the majority direction. On 30s crypto this
    sits near 0.50, which is exactly why direction accuracy alone flatters
    a model and must be shown against this number.
    """
    up = float(np.mean(actual > 0))
    return max(up, 1.0 - up)


def net_edge_bps(
    pred: np.ndarray,
    raw_returns: np.ndarray,
    threshold_pct: float = 90.0,
) -> tuple:
    """
    Trade only the most confident predictions, then subtract costs.

    Returns (net basis points per trade, number of trades). Negative means
    the signal does not pay for its own fees, which is the usual outcome
    and worth seeing plainly.
    """
    if len(pred) < 20:
        return 0.0, 0
    conf = np.abs(pred)
    cut = np.percentile(conf, threshold_pct)
    mask = conf >= cut
    if mask.sum() == 0:
        return 0.0, 0

    direction = np.sign(pred[mask])
    gross_bps = direction * raw_returns[mask] * 10_000.0
    cost_bps = 2.0 * (TAKER_FEE_BPS + SLIPPAGE_BPS)
    net = gross_bps - cost_bps
    return float(np.mean(net)), int(mask.sum())


def walk_forward_splits(
    timestamps: np.ndarray,
    n_folds: int = 5,
    horizon_sec: int = 30,
    embargo_multiplier: int = 3,
):
    """
    Expanding-window splits with an embargo gap.

    The gap matters. A sample right before the test boundary has a label
    that reads prices inside the test set, so training on it leaks. The
    embargo drops horizon * multiplier seconds either side of every
    boundary. Without this the scores come out optimistic and wrong.
    """
    n = len(timestamps)
    if n < 200:
        return []

    embargo_ms = horizon_sec * embargo_multiplier * 1000
    fold_size = n // (n_folds + 1)
    splits = []

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_start = train_end
        test_end = min(fold_size * (k + 1), n)
        if test_end - test_start < 50:
            continue

        boundary_t = timestamps[test_start]
        # Drop the tail of train whose labels reach into test.
        safe_train = np.where(timestamps < boundary_t - embargo_ms)[0]
        if len(safe_train) < 100:
            continue
        test_idx = np.arange(test_start, test_end)
        splits.append((safe_train, test_idx))

    return splits


def evaluate_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    raw_returns: np.ndarray,
    feature_names: List[str],
    fit_predict_fn,
    n_folds: int = 5,
    horizon_sec: int = 30,
    threshold_pct: float = 90.0,
) -> EvalReport:
    """
    Run the walk-forward. fit_predict_fn(X_train, y_train, X_test) -> preds,
    so this file stays independent of which model is being scored.
    """
    report = EvalReport(horizon_sec=horizon_sec, threshold_pct=threshold_pct)
    splits = walk_forward_splits(timestamps, n_folds, horizon_sec)

    for i, (tr, te) in enumerate(splits):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        rr_te = raw_returns[te]

        try:
            preds = fit_predict_fn(X_tr, y_tr, X_te)
        except Exception as e:
            print(f"  fold {i}: model failed ({e}), skipping")
            continue

        preds = np.asarray(preds, dtype=np.float64).ravel()
        if len(preds) != len(y_te):
            print(f"  fold {i}: prediction length mismatch, skipping")
            continue

        bases = baseline_predictions(X_te, feature_names)
        base_ics = {k: information_coefficient(v, y_te) for k, v in bases.items()}

        conf = np.abs(preds)
        cut = np.percentile(conf, threshold_pct) if len(conf) else 0.0
        top = conf >= cut
        top_hit = direction_accuracy(preds[top], y_te[top]) if top.sum() else 0.5

        net_bps, n_trades = net_edge_bps(preds, rr_te, threshold_pct)

        report.folds.append(
            FoldResult(
                fold=i,
                n_train=len(tr),
                n_test=len(te),
                ic=information_coefficient(preds, y_te),
                ic_baselines=base_ics,
                top_decile_hit=top_hit,
                base_rate=majority_base_rate(y_te),
                net_bps_per_trade=net_bps,
                n_trades=n_trades,
            )
        )

    return report


def print_report(report: EvalReport, model_name: str = "model"):
    """Print the scoreboard, then say out loud whether the model earned its keep."""
    if not report.folds:
        print("No folds completed. Not enough data to score anything.")
        return

    print()
    print(f"Walk-forward evaluation: {model_name}")
    print(f"Horizon {report.horizon_sec}s | trading top {100 - report.threshold_pct:.0f}% by confidence")
    print(f"Costs: {TAKER_FEE_BPS}bp taker x2 + {SLIPPAGE_BPS}bp slippage x2 = {2*(TAKER_FEE_BPS+SLIPPAGE_BPS):.0f}bp round trip")
    print()
    print(f"{'fold':>4} {'train':>7} {'test':>6} {'IC':>7} {'IC_imb':>7} {'IC_mom':>7} {'hit%':>6} {'base%':>6} {'net_bp':>8} {'trades':>7}")
    print("-" * 78)
    for f in report.folds:
        print(
            f"{f.fold:>4} {f.n_train:>7} {f.n_test:>6} {f.ic:>7.4f} "
            f"{f.ic_baselines.get('imbalance', 0):>7.4f} {f.ic_baselines.get('momentum', 0):>7.4f} "
            f"{f.top_decile_hit*100:>6.2f} {f.base_rate*100:>6.2f} {f.net_bps_per_trade:>8.2f} {f.n_trades:>7}"
        )

    s = report.summary()
    print("-" * 78)
    print(f"IC mean {s['ic_mean']:+.4f} (sd {s['ic_std']:.4f}), positive in {s['ic_positive_folds']}/{s['n_folds']} folds")

    # Overlapping labels are the trap at long horizons. Snapshots are one
    # second apart, so at a 600s horizon consecutive labels share 599 of
    # their 600 seconds. Thousands of rows can carry only a handful of
    # independent observations, and a single lucky window then drives the
    # whole average.
    total_test = sum(f.n_test for f in report.folds)
    independent = total_test / max(1, report.horizon_sec)
    if independent < 100:
        print(f"WARNING: ~{independent:.0f} independent windows across all folds "
              f"({total_test} rows / {report.horizon_sec}s horizon).")
        print("         Labels overlap heavily. Treat every number above as provisional;")
        print("         one lucky window can carry the entire average.")
    print(f"Top-decile direction {s['top_decile_hit_mean']*100:.2f}% vs base {s['base_rate_mean']*100:.2f}% "
          f"(edge {s['edge_over_base']*100:+.2f}pp)")
    print(f"Net after fees: {s['net_bps_mean']:+.2f} bp/trade over {s['total_trades']} trades")

    base_ic = float(np.mean([f.ic_baselines.get("imbalance", 0.0) for f in report.folds]))
    print()
    print("Verdict:")
    if s["ic_mean"] > 0.15:
        print(f"  IC above 0.15 at a {report.horizon_sec}s horizon. Treat as a leak, not a discovery.")
        print("  Re-run predict/test_features.py and check the recording for gaps.")
    elif s["ic_mean"] <= 0:
        print("  No signal. Predictions are uncorrelated or backwards.")
    elif s["ic_positive_folds"] < s["n_folds"] * 0.6:
        print(f"  Unstable: positive in only {s['ic_positive_folds']}/{s['n_folds']} folds. Not tradeable.")
    elif s["ic_mean"] <= base_ic:
        print(f"  Model IC {s['ic_mean']:.4f} does not beat book imbalance alone ({base_ic:.4f}).")
        print("  Use the imbalance feature directly and drop the model.")
    elif s["net_bps_mean"] <= 0:
        print(f"  Real but too small: {s['net_bps_mean']:+.2f} bp/trade after costs.")
        print("  Usable as a display signal, not as a trade trigger.")
    else:
        profitable_folds = sum(1 for f in report.folds if f.net_bps_per_trade > 0)
        if profitable_folds <= len(report.folds) / 2:
            print(f"  Average is positive ({s['net_bps_mean']:+.2f} bp/trade) but only "
                  f"{profitable_folds}/{len(report.folds)} folds actually made money.")
            print("  One lucky fold is carrying the mean. Not a strategy.")
        else:
            print(f"  Signal beats baselines and covers costs at {s['net_bps_mean']:+.2f} bp/trade")
            print(f"  in {profitable_folds}/{len(report.folds)} folds.")
            print("  Confirm on out-of-sample days before trusting it.")
    print()


def demo():
    """Self-check: the scoreboard must find signal when it exists and none when it doesn't."""
    rng = np.random.default_rng(0)
    n = 4000
    names = ["book_imbalance", "ret_30s", "noise"]

    ts = np.arange(n, dtype=np.int64) * 1000
    X = rng.normal(size=(n, 3)).astype(np.float32)
    # A modest planted edge. Strong enough that a working scoreboard must
    # find it, weak enough that a broken one plausibly would not.
    y = (0.15 * X[:, 0] + rng.normal(size=n)).astype(np.float32)
    raw = (y * 0.0004).astype(np.float64)

    def fit_pred(Xtr, ytr, Xte):
        # Least squares with an intercept, via numpy, so this check needs no ML deps.
        A = np.hstack([Xtr.astype(np.float64), np.ones((len(Xtr), 1))])
        coef, *_ = np.linalg.lstsq(A, ytr.astype(np.float64), rcond=None)
        B = np.hstack([Xte.astype(np.float64), np.ones((len(Xte), 1))])
        return B @ coef

    rep = evaluate_walk_forward(X, y, ts, raw, names, fit_pred, n_folds=4)
    s = rep.summary()
    assert s["n_folds"] >= 3, "walk-forward produced too few folds"
    assert s["ic_mean"] > 0.01, f"failed to detect a planted edge (IC {s['ic_mean']:.4f})"
    print(f"detects planted edge: IC {s['ic_mean']:+.4f}")

    # Pure noise must score ~0. A scoreboard that finds signal in noise is worthless.
    y_noise = rng.normal(size=n).astype(np.float32)
    rep2 = evaluate_walk_forward(X, y_noise, ts, raw, names, fit_pred, n_folds=4)
    ic2 = rep2.summary()["ic_mean"]
    assert abs(ic2) < 0.06, f"found signal in pure noise (IC {ic2:.4f})"
    print(f"reports no signal on noise: IC {ic2:+.4f}")

    # Embargo must actually remove samples near the boundary.
    splits = walk_forward_splits(ts, n_folds=4, horizon_sec=30)
    for tr, te in splits:
        gap_ms = ts[te[0]] - ts[tr[-1]]
        assert gap_ms >= 30 * 3 * 1000, f"embargo gap too small: {gap_ms}ms"
    print(f"embargo enforced on {len(splits)} splits")
    print("evaluate.py self-check passed")


if __name__ == "__main__":
    demo()
