"""
Evaluate, then train and save.

    python -m forecast.train --symbol BTCUSDT --horizon 60
    python -m forecast.train --symbol BTCUSDT --horizon 60 --eval-only

Walk-forward runs first and decides what ships. A component that does not beat
its baseline out of sample is not saved, and the served forecast says so
instead of showing a number with nothing behind it.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forecast.data import load_history
from forecast.direction import DirectionModel
from forecast.evaluate import calibrate, make_model, summarize, walk_forward
from forecast.features import FEATURES, dataset

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models")) / "forecast"


def gate(res) -> dict:
    n = len(res)
    vol_ok = bool((res.vol_r2 > res.vol_r2_har).sum() > n / 2 and res.vol_r2.mean() > 0)
    calls = res.calls.sum()
    hit = (res.hit * res.calls).sum() / calls
    up = (res.hit_always_up * res.calls).sum() / calls
    # Calls must be right clearly more often than blindly guessing "up", and
    # not just on average: in at least two quarters out of three.
    dir_ok = bool(hit > 0.52 and hit > up + 0.02 and (res.hit > res.hit_always_up).sum() >= int(np.ceil(2 * n / 3)))
    return {"vol": vol_ok, "dir": dir_ok}


def _save_atomic(model, path: Path):
    tmp = path.with_suffix(".tmp.json")
    model.save_model(tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--horizon", type=int, default=60, help="minutes")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--eval-only", action="store_true")
    # XAUUSDT listed 2025-12, too short for 12-month training folds.
    ap.add_argument("--min-train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--oos-out", help="pickle out-of-sample predictions + labels here, for analysis")
    a = ap.parse_args()
    sym, h = a.symbol.upper(), a.horizon

    df = load_history(sym)
    X, y = dataset(df, h, sym)
    print(f"{sym} {h}m: {len(X):,} rows {X.index[0].date()} .. {X.index[-1].date()}")
    res, oos = walk_forward(X, y, h, stride=a.stride, n_estimators=a.trees,
                            min_train_months=a.min_train_months, test_months=a.test_months)
    print(summarize(res))
    if a.oos_out:
        oos.join(y).join(X).to_pickle(a.oos_out)
    cal = calibrate(oos, y, h, res)
    print(f"calibration: k80 {cal['k80']:.3f}  floor/ceiling z {cal['zq']['0.1']:+.3f}/{cal['zq']['0.9']:+.3f}")
    ship = gate(res)
    print(f"ship: vol={ship['vol']} dir={ship['dir']}")
    if a.eval_only:
        return
    if not ship["vol"]:
        print("Vol model did not beat HAR out of sample. Nothing saved.")
        return

    idx = np.arange(0, len(X), a.stride)
    out = MODEL_DIR / f"{sym}_{h}m"
    out.mkdir(parents=True, exist_ok=True)
    _save_atomic(make_model(a.trees).fit(X.iloc[idx], y["lvol_rel"].iloc[idx]), out / "vol.json")
    if ship["dir"]:
        (out / "dir.tmp").write_text(DirectionModel.fit(X.iloc[idx], y["ret_z"].iloc[idx]).to_json())
        os.replace(out / "dir.tmp", out / "dir.json")
    elif (out / "dir.json").exists():
        (out / "dir.json").unlink()  # a stale direction model must not outlive a failed gate

    m = res.mean(numeric_only=True)
    meta = {
        "symbol": sym, "horizon_min": h, "features": FEATURES, "ship": ship,
        "k80": cal["k80"], "zq": cal["zq"],
        "trained_through": str(X.index[-1]), "trained_at": datetime.now(timezone.utc).isoformat(),
        "oos": {"folds": len(res), "vol_r2": m.vol_r2, "vol_r2_har": m.vol_r2_har, "vol_ic": m.vol_ic,
                "dir_calls": cal["calls"], "dir_hit": cal["hit"],
                "dir_hit_always_up": float((res.hit_always_up * res.calls).sum() / res.calls.sum()),
                "dir_folds_beating_up": int((res.hit > res.hit_always_up).sum()), "dir_worst_fold": float(res.hit.min()),
                "dir_win_bp": cal["win_bp"], "dir_loss_bp": cal["loss_bp"]},
    }
    tmp = out / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=float))
    os.replace(tmp, out / "meta.json")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
