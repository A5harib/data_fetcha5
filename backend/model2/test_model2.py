"""
Self-checks for model 2. Uses the cached spot history, no network.

    python -m model2.test_model2
"""
from __future__ import annotations

import json

import numpy as np

from model2.agent import WARMUP, Agent, advance, features
from model2.bot import FEED_BARS
from model2.data import load


def _slice(sym="BTCUSDT", n=WARMUP + 40 * 1440):
    df = load(sym)
    return df.iloc[200_000:200_000 + n]


def test_features_use_only_the_past(df):
    full = features(df, "BTCUSDT")
    for cut in (WARMUP + 100, WARMUP + 5000, len(df) - 1):
        part = features(df.iloc[:cut + 1], "BTCUSDT")
        a, b = full.iloc[cut].to_numpy(float), part.iloc[-1].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), f"row {cut} changed when the future was removed"
    print("ok  features use only the past")


def test_live_matches_replay(df):
    """
    The live loop sees a sliding window of recent bars, a little at a time, and
    may restart from a saved copy. It must end where one long replay ends.
    """
    X = features(df, "BTCUSDT").to_numpy(float)
    rows = np.flatnonzero(np.isfinite(X).all(axis=1))[: 7 * 1440]
    scaler = Agent.fit_scaler(X[rows])

    one = Agent("BTCUSDT", scaler)
    one.start_real(100.0, "test")
    advance(one, df)

    inc = Agent("BTCUSDT", scaler)
    inc.start_real(100.0, "test")
    step = 997  # not a multiple of 60, so hour boundaries land mid-chunk
    end = WARMUP + 1
    restarted = False
    while end <= len(df):
        win = df.iloc[max(0, end - FEED_BARS):end]
        advance(inc, win)
        if not restarted and end > len(df) // 2:
            inc = Agent.from_dict(json.loads(inc.dumps()))  # save and reload, like a Space restart
            restarted = True
        if end == len(df):
            break
        end = min(end + step, len(df))

    assert one.hours_learned == inc.hours_learned > 0, (one.hours_learned, inc.hours_learned)
    assert np.allclose(one.w, inc.w, rtol=1e-6, atol=1e-9), np.abs(one.w - inc.w).max()
    a, b = one.real["trades"], inc.real["trades"]
    assert len(a) == len(b), (len(a), len(b))
    assert all(x["entry_t"] == y["entry_t"] and x["exit_t"] == y["exit_t"] for x, y in zip(a, b))
    assert abs(one.real["cash"] - inc.real["cash"]) < 1e-9
    print(f"ok  live loop with a restart matches the replay ({one.hours_learned} hours learned, {len(a)} paper trades)")


def test_learns_only_from_known_outcomes(df):
    """Every learned row's outcome window must have closed before the bar the agent acts on."""
    X = features(df, "BTCUSDT").to_numpy(float)
    rows = np.flatnonzero(np.isfinite(X).all(axis=1))[: 7 * 1440]
    ag = Agent("BTCUSDT", Agent.fit_scaler(X[rows]))
    seen = []
    orig = ag.learn

    def spy(Z, y):
        seen.append(ag.learned_up_to)
        orig(Z, y)
    ag.learn = spy
    last = []
    orig_step = ag.step

    def step_spy(t, z, p):
        last.append((t, ag.learned_up_to))
        return orig_step(t, z, p)
    ag.step = step_spy
    advance(ag, df)
    from model2.agent import H
    for t, up in last:
        if up is not None:
            assert up - 1 + H <= t - 1, f"bar {t} learned from a row whose outcome ends at {up - 1 + H}"
    print("ok  learns only from outcomes known before each decision")


if __name__ == "__main__":
    d = _slice()
    test_features_use_only_the_past(d)
    test_learns_only_from_known_outcomes(d)
    test_live_matches_replay(d)
    print("all model 2 checks passed")
