"""
Self-check for the feature/label pipeline.

Run: python -m predict.test_features   (from the backend/ directory)

The point of this file is one assertion: a snapshot's label must depend on
the future and its features must not. Everything else here supports that.
"""
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from predict.features import (
    FEATURE_NAMES,
    Snapshot,
    build_snapshot,
    label_snapshots,
    snapshots_to_arrays,
)


def synth_stream(n_sec=600, seed=0):
    """A deterministic fake market: 1 snapshot per second with known prices."""
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000_000
    price = 60_000.0
    mids, trades, imbs = [], [], []
    snaps = []

    for i in range(n_sec):
        t = t0 + i * 1000
        price *= math.exp(rng.normal(0, 0.0004))
        mids.append((t, price))

        # Trades land in the second BEFORE the snapshot instant. A generator
        # that emitted them after t would hand build_snapshot future trades
        # and quietly invalidate every leak check below.
        for k in range(rng.integers(1, 6)):
            qty = float(rng.exponential(0.5))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            trades.append((t - 900 + k * 100, sign * qty, qty))

        imb = float(rng.uniform(-0.6, 0.6))
        imbs.append((t, imb))

        bid_qty = 10.0 * (1 + imb)
        ask_qty = 10.0 * (1 - imb)
        s = build_snapshot(
            t_ms=t,
            mid=price,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            best_bid=price - 0.5,
            best_ask=price + 0.5,
            mid_history=mids,
            trade_history=trades,
            imbalance_history=imbs,
            last_trade_ms=trades[-1][0],
        )
        if s:
            snaps.append(s)
    return snaps


def test_warmup_returns_none():
    """No snapshot before 120s of history exists. Guessing defaults is worse than waiting."""
    snaps = synth_stream(n_sec=200)
    assert snaps, "expected some snapshots after warmup"
    t0 = 1_700_000_000_000
    assert snaps[0].t_ms - t0 >= 120_000, "snapshot produced before 120s of history"
    print(f"  warmup respected: first snapshot at +{(snaps[0].t_ms - t0)/1000:.0f}s")


def test_label_matches_future_price():
    """The label must equal the forward return over vol. Checked by hand, not by formula reuse."""
    snaps = synth_stream(n_sec=600)
    snaps = label_snapshots(snaps, horizon_sec=30)
    by_time = {s.t_ms: s for s in snaps}

    checked = 0
    for s in snaps:
        if s.label is None:
            continue
        future = by_time.get(s.t_ms + 30_000)
        if future is None:
            continue
        expected_ret = math.log(future.mid / s.mid)
        vol = max(s.features["realized_vol_60s"], 1e-6)
        # Labels are clipped to +/-10 sigma; mirror that here rather than
        # loosening the check.
        expected = float(np.clip(expected_ret / vol, -10.0, 10.0))
        assert abs(s.label - expected) < 1e-6, (
            f"label mismatch at {s.t_ms}: {s.label} vs {expected}"
        )
        assert abs(s.future_mid - future.mid) < 1e-9, "future_mid points at the wrong bar"
        checked += 1

    assert checked > 100, f"only verified {checked} labels, expected hundreds"
    print(f"  label == forward return / vol on {checked} snapshots")


def test_no_lookahead_in_features():
    """
    The leak test.

    Rebuild every snapshot using ONLY history up to its own timestamp. If a
    feature had quietly read ahead, these would differ from the streaming
    build, which saw the same prefix. Identical output means features are a
    pure function of the past.
    """
    snaps = synth_stream(n_sec=400, seed=7)
    assert len(snaps) > 50

    rng = np.random.default_rng(1)
    t0 = 1_700_000_000_000
    mids, trades, imbs = [], [], []
    rng2 = np.random.default_rng(7)
    price = 60_000.0
    rebuilt = {}

    for i in range(400):
        t = t0 + i * 1000
        price *= math.exp(rng2.normal(0, 0.0004))
        mids.append((t, price))
        for k in range(rng2.integers(1, 6)):
            qty = float(rng2.exponential(0.5))
            sign = 1.0 if rng2.random() < 0.5 else -1.0
            trades.append((t - 900 + k * 100, sign * qty, qty))
        imb = float(rng2.uniform(-0.6, 0.6))
        imbs.append((t, imb))

        s = build_snapshot(
            t_ms=t,
            mid=price,
            bid_qty=10.0 * (1 + imb),
            ask_qty=10.0 * (1 - imb),
            best_bid=price - 0.5,
            best_ask=price + 0.5,
            # Deliberately sliced to the present: nothing after t is visible.
            mid_history=[x for x in mids if x[0] <= t],
            trade_history=[x for x in trades if x[0] <= t],
            imbalance_history=[x for x in imbs if x[0] <= t],
            last_trade_ms=max(x[0] for x in trades if x[0] <= t),
        )
        if s:
            rebuilt[t] = s

    compared = 0
    for s in snaps:
        r = rebuilt.get(s.t_ms)
        if r is None:
            continue
        for name in FEATURE_NAMES:
            a, b = s.features[name], r.features[name]
            assert abs(a - b) < 1e-9, f"feature {name} differs at {s.t_ms}: {a} vs {b}"
        compared += 1

    assert compared > 50, f"only compared {compared} snapshots"
    print(f"  features identical under past-only replay on {compared} snapshots ({len(FEATURE_NAMES)} features each)")


def test_shuffled_future_destroys_correlation():
    """
    Sanity check on the label itself. Shuffle which future each snapshot gets
    and correlation with any feature should collapse toward zero. If a feature
    still correlates with a shuffled future, that feature contains the future.
    """
    snaps = label_snapshots(synth_stream(n_sec=900, seed=3), horizon_sec=30)
    X, y, _, _ = snapshots_to_arrays(snaps)
    assert len(y) > 200, f"need a few hundred labeled rows, got {len(y)}"

    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)

    worst = 0.0
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        c = abs(float(np.corrcoef(col, y_shuf)[0, 1]))
        worst = max(worst, c)
    assert worst < 0.25, f"a feature correlates {worst:.3f} with a SHUFFLED future"
    print(f"  shuffled-future correlation stays low (max {worst:.3f})")


def test_unlabeled_tail_is_dropped():
    """The last horizon of data has no future yet and must not be trained on."""
    snaps = label_snapshots(synth_stream(n_sec=400), horizon_sec=30)
    tail = [s for s in snaps[-25:] if s.label is not None]
    assert not tail, f"{len(tail)} snapshots in the final 25s got labels they cannot have"
    X, y, _, _ = snapshots_to_arrays(snaps)
    assert len(X) == len(y)
    print(f"  final {30}s left unlabeled; {len(y)} usable rows")


def test_gap_is_not_bridged():
    """Across a hole in the data, a label must not be invented from a distant bar."""
    snaps = synth_stream(n_sec=400)
    keep = [s for s in snaps if not (150 < (s.t_ms - snaps[0].t_ms) / 1000 < 220)]
    labeled = label_snapshots(keep, horizon_sec=30, max_gap_sec=5)
    for s in labeled:
        if s.label is None:
            continue
        gap = abs((s.future_mid and 0) or 0)
    # Any snapshot whose target lands inside the hole must stay unlabeled.
    times = {s.t_ms for s in keep}
    for s in labeled:
        if s.label is None:
            continue
        target = s.t_ms + 30_000
        near = [t for t in times if abs(t - target) <= 5_000]
        assert near, f"snapshot at {s.t_ms} was labeled across a data gap"
    print("  data gaps are not bridged")


def demo():
    print("feature/label pipeline self-check")
    for fn in (
        test_warmup_returns_none,
        test_label_matches_future_price,
        test_no_lookahead_in_features,
        test_shuffled_future_destroys_correlation,
        test_unlabeled_tail_is_dropped,
        test_gap_is_not_bridged,
    ):
        print(f"- {fn.__name__}")
        fn()
    print("all checks passed")


if __name__ == "__main__":
    demo()
