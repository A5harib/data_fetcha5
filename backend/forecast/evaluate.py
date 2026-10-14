"""
Walk-forward evaluation. Expanding train window, 3-month test blocks, and an
embargo of one horizon between them so no training label overlaps the test
period.

Two questions, answered separately:
  vol        Does the model forecast realized volatility better than the
             obvious baselines (reference vol alone, last hour's vol)?
  direction  When the snap-back rule makes a call, is it right more often
             than "always up", in most quarters? And what does a call earn?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from forecast.direction import DirectionModel

# Round trip, basis points. Taker: 5bp a side plus 1bp slippage a side.
# Maker: 2bp a side, no slippage, and assumes every limit order fills, which
# flatters it.
COST_TAKER_BP = 12.0
COST_MAKER_BP = 4.0


def make_model(n_estimators: int = 300) -> xgb.XGBRegressor:
    """Volatility model. Shallow trees with large leaves: plenty of signal, still noisy targets."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=200,
        reg_lambda=10.0,
        tree_method="hist",
        n_jobs=-1,
    )


def folds(index: pd.DatetimeIndex, horizon: int, min_train_months: int = 12, test_months: int = 3):
    start = index[0] + pd.DateOffset(months=min_train_months)
    end = index[-1]
    t0 = start
    while t0 < end:
        t1 = min(t0 + pd.DateOffset(months=test_months), end + pd.Timedelta(minutes=1))
        train = index < t0 - pd.Timedelta(minutes=horizon + 1)
        test = (index >= t0) & (index < t1)
        if test.sum() > horizon * 10:
            yield t0, train, test
        t0 = t1


def _ic(a, b) -> float:
    return float(spearmanr(a, b)[0])


def decisions(calls: np.ndarray, horizon: int) -> np.ndarray:
    """
    Positions of the calls a user would actually act on: take a call, then
    ignore new ones until it resolves `horizon` bars later. Keeps every scored
    call independent, so hit rates are not inflated by counting one move 60 times.
    """
    out, i, n = [], 0, len(calls)
    while i < n:
        if calls[i] != 0:
            out.append(i)
            i += horizon
        else:
            i += 1
    return np.asarray(out, dtype=int)


def walk_forward(X: pd.DataFrame, y: pd.DataFrame, horizon: int, stride: int = 5,
                 n_estimators: int = 300, verbose: bool = True,
                 min_train_months: int = 12, test_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, out-of-sample predictions for every test row)."""
    rows, oos = [], []
    for t0, tr, te in folds(X.index, horizon, min_train_months, test_months):
        tr_idx = np.flatnonzero(tr)[::stride]  # labels overlap; thinning loses little
        Xtr, Xte = X.iloc[tr_idx], X[te]
        yt = y[te]

        vol_m = make_model(n_estimators).fit(Xtr, y["lvol_rel"].iloc[tr_idx])
        p_vol = vol_m.predict(Xte)
        # Vol baselines. sigma_ref alone predicts lvol_rel = 0. Last hour's vol
        # predicts lrv_60. A linear HAR fit on the same train rows is the
        # textbook model a tree has to beat to justify itself.
        har_cols = ["lrv_15", "lrv_60", "lrv_240", "lrv_1440", "hour_sin", "hour_cos", "weekend"]
        A = np.c_[np.ones(len(tr_idx)), Xtr[har_cols].to_numpy(float)]
        coef = np.linalg.lstsq(A, y["lvol_rel"].iloc[tr_idx].to_numpy(float), rcond=None)[0]
        p_har = np.c_[np.ones(len(Xte)), Xte[har_cols].to_numpy(float)] @ coef
        v = yt["lvol_rel"].to_numpy()
        sst = np.sum((v - v.mean()) ** 2)
        r2 = lambda p: 1 - np.sum((v - p) ** 2) / sst  # noqa: E731

        dir_m = DirectionModel.fit(Xtr, y["ret_z"].iloc[tr_idx])
        score, call = dir_m.score(Xte), dir_m.call(Xte)
        k = decisions(call, horizon)
        r = yt["ret"].to_numpy()[k]
        pnl = call[k] * r * 1e4
        hit = np.sign(r) == call[k]

        rows.append({
            "test_start": t0.date(), "n_train": len(tr_idx), "n_test": int(te.sum()),
            "vol_r2": r2(p_vol), "vol_r2_har": r2(p_har), "vol_r2_last60": r2(Xte["lrv_60"].to_numpy()),
            "vol_ic": _ic(p_vol, v),
            "dir_ic": _ic(score, yt["ret_z"]), "calls": len(k),
            "hit": float(hit.mean()) if len(k) else np.nan,
            "hit_always_up": float((r > 0).mean()) if len(k) else np.nan,
            "win_bp": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else np.nan,
            "loss_bp": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else np.nan,
            "gross_bp": float(pnl.mean()) if len(k) else np.nan,
        })
        oos.append(pd.DataFrame({"p_vol": p_vol, "score": score, "call": call}, index=Xte.index))
        if verbose:
            m = rows[-1]
            print(f"  {m['test_start']}  vol R2 {m['vol_r2']:+.3f} (HAR {m['vol_r2_har']:+.3f})  "
                  f"calls {m['calls']:4d}  right {m['hit']:.1%} (always-up {m['hit_always_up']:.1%})  "
                  f"avg {m['gross_bp']:+6.2f}bp", flush=True)
    return pd.DataFrame(rows), pd.concat(oos)


def calibrate(oos: pd.DataFrame, y: pd.DataFrame, horizon: int, res: pd.DataFrame) -> dict:
    """
    Turn raw OOS predictions into numbers that mean what they say.
      k80    band multiplier so that 80% of realized returns land inside
             +/- k80 * predicted sigma over the horizon
      zq     signed quantiles of the standardized move (floor/ceiling)
      hit    pooled share of independent direction calls that were right
    """
    yy = y.loc[oos.index]
    sig_h = np.exp(oos["p_vol"].to_numpy()) * yy["sigma_ref"].to_numpy() * np.sqrt(horizon)
    z = yy["ret"].to_numpy() / sig_h
    k80 = float(np.quantile(np.abs(z), 0.8))
    # One-sided "9 in 10" floor and ceiling, and room for skew if a horizon is lopsided.
    zq = {str(q): float(np.quantile(z, q)) for q in (0.05, 0.1, 0.9, 0.95)}
    n = res["calls"].sum()
    hit = float((res["hit"] * res["calls"]).sum() / n)
    win = float((res["win_bp"] * res["calls"]).sum() / n)
    loss = float((res["loss_bp"] * res["calls"]).sum() / n)
    return {"k80": k80, "zq": zq, "hit": hit, "calls": int(n), "win_bp": win, "loss_bp": loss}


def summarize(res: pd.DataFrame) -> str:
    m = res.mean(numeric_only=True)
    n = res.calls.sum()
    hit = (res.hit * res.calls).sum() / n
    up = (res.hit_always_up * res.calls).sum() / n
    return "\n".join([
        f"folds {len(res)}",
        f"vol   R2 {m.vol_r2:+.3f}   HAR {m.vol_r2_har:+.3f}   last-60m {m.vol_r2_last60:+.3f}   "
        f"IC {m.vol_ic:+.3f}   model beats HAR in {(res.vol_r2 > res.vol_r2_har).sum()}/{len(res)} folds",
        f"dir   {int(n)} independent calls, right {hit:.1%} vs always-up {up:.1%}, "
        f"beats always-up in {(res.hit > res.hit_always_up).sum()}/{len(res)} folds, worst fold {res.hit.min():.1%}",
        f"      avg win {m.win_bp:+.1f}bp  avg loss {m.loss_bp:+.1f}bp  avg per call {m.gross_bp:+.2f}bp "
        f"(costs: taker {COST_TAKER_BP:.0f}bp, maker {COST_MAKER_BP:.0f}bp round trip)",
    ])
