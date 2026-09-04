"""
End-to-end check: recorder -> features -> labels -> train -> score.

Run: python -m predict.test_pipeline   (from backend/)

Two synthetic markets are fed through the real code path. In one, order book
imbalance genuinely predicts the next move. In the other, price is a pure
random walk. The pipeline must find the edge in the first and report nothing
in the second. A pipeline that cannot tell those apart is worthless no matter
how good its numbers look.
"""
import math
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from predict.evaluate import evaluate_walk_forward
from predict.features import FEATURE_NAMES, label_snapshots, snapshots_to_arrays

from predict.recorder import MarketRecorder
from predict.train import fit_once, load_snapshots


def simulate(n_sec: int, edge: float, seed: int = 0, out_path: Path = None):
    """
    Drive the real MarketRecorder with a synthetic feed.

    edge > 0 means book imbalance at time t genuinely pushes the price over
    the following seconds. edge == 0 is a pure random walk with a book that
    tells you nothing.
    """
    rng = np.random.default_rng(seed)
    rec = MarketRecorder(symbol="TEST", out_path=out_path)
    rec.open()

    t0 = 1_700_000_000_000
    price = 60_000.0
    # Imbalance is persistent, so its effect can act over a 30s horizon
    # rather than being averaged away second by second.
    imb = 0.0
    pending = []
    snaps = []

    for i in range(n_sec):
        t = t0 + i * 1000
        imb = 0.92 * imb + 0.08 * float(rng.normal(0, 0.35))
        imb = float(np.clip(imb, -0.95, 0.95))

        # The planted edge: today's imbalance nudges the next 30s of drift.
        pending.append(edge * imb)
        if len(pending) > 30:
            pending.pop(0)
        drift = sum(pending) / 30.0 if pending else 0.0

        price *= math.exp(drift + rng.normal(0, 0.00035))

        # Trades arrive strictly before the snapshot instant.
        for k in range(int(rng.integers(2, 7))):
            qty = float(rng.exponential(0.4)) + 0.01
            buyer_maker = rng.random() > (0.5 + 0.3 * imb)
            rec.on_trade(t_ms=t - 900 + k * 120, price=price, qty=qty, is_buyer_maker=buyer_maker)

        depth = 12.0
        bid_qty = depth * (1.0 + imb)
        ask_qty = depth * (1.0 - imb)
        half_spread = price * 0.00002
        rec.on_depth(
            t_ms=t,
            bids=[[price - half_spread, bid_qty / 5]] * 5,
            asks=[[price + half_spread, ask_qty / 5]] * 5,
        )
        # Force the recorder's own imbalance to reflect the simulated book.
        rec.bid_qty, rec.ask_qty = bid_qty, ask_qty

        s = rec.tick(now_ms=t)
        if s:
            snaps.append(s)

    rec.close()
    return snaps


def score(snaps, horizon=30, folds=4):
    snaps = label_snapshots(snaps, horizon_sec=horizon)
    X, y, ts, raw = snapshots_to_arrays(snaps)
    if len(y) < 600:
        return None, len(y)

    def fit_predict(X_tr, y_tr, X_te):
        n = len(X_tr)
        core, sc = fit_once(X_tr, y_tr, ts[:n], kind="ridge", now_ms=int(ts[n - 1]))
        return core.predict(sc.transform(X_te))

    rep = evaluate_walk_forward(
        X=X, y=y, timestamps=ts, raw_returns=raw,
        feature_names=FEATURE_NAMES, fit_predict_fn=fit_predict,
        n_folds=folds, horizon_sec=horizon,
    )
    return rep.summary(), len(y)


def test_finds_planted_edge():
    """With a real relationship in the data, the pipeline must report positive IC."""
    snaps = simulate(n_sec=5000, edge=0.0015, seed=1)
    s, n = score(snaps)
    assert s, f"not enough labeled samples: {n}"
    assert s["ic_mean"] > 0.05, f"missed a planted edge (IC {s['ic_mean']:.4f})"
    assert s["ic_positive_folds"] >= s["n_folds"] - 1, "edge not stable across folds"
    print(f"  finds planted edge: IC {s['ic_mean']:+.4f} across {s['n_folds']} folds ({n} samples)")


def test_reports_nothing_on_random_walk():
    """
    The important half. On an unpredictable market the pipeline must come
    back near zero. Anything else means something is leaking.
    """
    snaps = simulate(n_sec=5000, edge=0.0, seed=2)
    s, n = score(snaps)
    assert s, f"not enough labeled samples: {n}"
    assert abs(s["ic_mean"]) < 0.06, f"found signal in a random walk (IC {s['ic_mean']:+.4f}) -- leak"
    print(f"  random walk gives no signal: IC {s['ic_mean']:+.4f} ({n} samples)")


def test_jsonl_roundtrip():
    """What the recorder writes must load back byte-identical in feature values."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "TEST_snapshots.jsonl"
        written = simulate(n_sec=800, edge=0.001, seed=3, out_path=p)
        assert p.exists() and written, "recorder wrote nothing"

        loaded = load_snapshots(p)
        assert len(loaded) == len(written), f"wrote {len(written)}, read {len(loaded)}"
        for a, b in zip(written, loaded):
            assert a.t_ms == b.t_ms
            for name in FEATURE_NAMES:
                assert abs(a.features[name] - b.features[name]) < 1e-9, f"{name} changed on disk"
        print(f"  {len(loaded)} snapshots survive the disk round-trip unchanged")


def test_horizon_lengthens_signal():
    """
    A slow-moving edge should be easier to see at 30s than at 5s. This
    catches a horizon parameter that is silently ignored.
    """
    snaps = simulate(n_sec=5000, edge=0.0015, seed=4)
    s5, _ = score(snaps, horizon=5)
    s30, _ = score(snaps, horizon=30)
    assert s5 and s30
    assert s30["ic_mean"] > s5["ic_mean"], (
        f"horizon has no effect: 5s IC {s5['ic_mean']:.4f}, 30s IC {s30['ic_mean']:.4f}"
    )
    print(f"  horizon matters: 5s IC {s5['ic_mean']:+.4f} -> 30s IC {s30['ic_mean']:+.4f}")


def test_shuffled_labels_kill_signal():
    """Break the pairing between features and future, and the edge must vanish."""
    snaps = label_snapshots(simulate(n_sec=5000, edge=0.0015, seed=5), horizon_sec=30)
    X, y, ts, raw = snapshots_to_arrays(snaps)
    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)

    def fit_predict(X_tr, y_tr, X_te):
        n = len(X_tr)
        core, sc = fit_once(X_tr, y_tr, ts[:n], kind="ridge", now_ms=int(ts[n - 1]))
        return core.predict(sc.transform(X_te))

    rep = evaluate_walk_forward(
        X=X, y=y_shuf, timestamps=ts, raw_returns=raw,
        feature_names=FEATURE_NAMES, fit_predict_fn=fit_predict,
        n_folds=4, horizon_sec=30,
    )
    ic = rep.summary()["ic_mean"]
    assert abs(ic) < 0.06, f"shuffled labels still score IC {ic:+.4f} -- leak"
    print(f"  shuffled labels destroy the edge: IC {ic:+.4f}")


def test_depth_without_trades_produces_nothing():
    """
    Regression test for a bug that cost real recording time.

    Binance's @aggTrade stream returns no messages on the futures combined
    endpoint, while @depth5 works fine. The recorder therefore received a
    healthy-looking feed, built zero snapshots, and reported no error at all.

    This pins the behaviour: depth alone must produce nothing, so a caller
    seeing zero snapshots knows trades are missing rather than assuming the
    market is quiet.
    """
    rec = MarketRecorder(symbol="TEST")
    t0 = 1_700_000_000_000
    price = 60_000.0
    for i in range(400):
        t = t0 + i * 1000
        rec.on_depth(
            t_ms=t,
            bids=[[price - 1, 5.0]] * 5,
            asks=[[price + 1, 5.0]] * 5,
        )
        rec.tick(now_ms=t)
    assert rec.snapshots_written == 0, (
        f"built {rec.snapshots_written} snapshots with no trade data; "
        "features would be fabricated"
    )
    assert len(rec.trade_history) == 0
    assert len(rec.mid_history) > 100, "depth was not ingested at all"
    print("  depth without trades yields 0 snapshots (no fabricated features)")


def demo():
    print("end-to-end pipeline check")
    for fn in (
        test_depth_without_trades_produces_nothing,
        test_jsonl_roundtrip,
        test_finds_planted_edge,
        test_reports_nothing_on_random_walk,
        test_horizon_lengthens_signal,
        test_shuffled_labels_kill_signal,
    ):
        print(f"- {fn.__name__}")
        fn()
    print("all checks passed")


if __name__ == "__main__":
    demo()
