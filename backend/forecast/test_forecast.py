"""
Self-checks. No network, no data files.

    python -m forecast.test_forecast
"""
import numpy as np
import pandas as pd

from forecast.data import KLINE_COLS, assemble
from forecast.features import FEATURES, WARMUP, build_features, make_labels


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = 1_700_000_000_000 + np.arange(n) * 60_000
    c = 30_000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    o = np.r_[c[0], c[:-1]]
    vol = rng.gamma(2, 10, n)
    k = pd.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) * 1.0003,
                      "low": np.minimum(o, c) * 0.9997, "close": c, "volume": vol,
                      "close_time": t + 59_999, "quote_volume": vol * c,
                      "count": rng.integers(50, 500, n), "taker_buy_volume": vol * rng.uniform(0.2, 0.8, n),
                      "taker_buy_quote_volume": 0, "ignore": 0})[KLINE_COLS]
    p = k.copy()
    p["close"] = rng.normal(0, 3e-4, n)
    f = pd.DataFrame({"t": t[::480] + 5, "funding": rng.normal(1e-4, 5e-5, len(t[::480]))})
    return assemble(k, p, f)


def test_features_are_past_only():
    df = synthetic(WARMUP + 1500)
    full = build_features(df)
    for cut in (WARMUP + 10, WARMUP + 700, len(df) - 1):
        part = build_features(df.iloc[: cut + 1])
        a, b = full.iloc[cut], part.iloc[-1]
        assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-5, atol=1e-6, equal_nan=True), (
            f"row {cut}: features change when the future is removed: "
            f"{[f for f in FEATURES if not np.isclose(a[f], b[f], rtol=1e-5, atol=1e-6, equal_nan=True)]}")
    # Scrambling every bar after t must not move features at t either.
    t = WARMUP + 300
    df2 = df.copy()
    df2.iloc[t + 1:, df2.columns.get_loc("close")] *= 1.5
    df2.iloc[t + 1:, df2.columns.get_loc("taker_buy_volume")] = 0
    assert np.allclose(build_features(df2).iloc[t], full.iloc[t], rtol=1e-5, atol=1e-6, equal_nan=True)


def test_labels_read_exactly_the_window():
    h = 15
    df = synthetic(500)
    y = make_labels(df, h)
    t = 300  # past the reference-vol warmup
    c = df["close"].to_numpy()
    assert np.isclose(y["ret"].iloc[t], np.log(c[t + h] / c[t]))
    # A bar beyond t+h must not touch the label at t; bar t+1 must.
    far, near = df.copy(), df.copy()
    far.iloc[t + h + 1, far.columns.get_loc("close")] *= 1.1
    near.iloc[t + 1, near.columns.get_loc("close")] *= 1.1
    assert np.isclose(make_labels(far, h)["lvol_rel"].iloc[t], y["lvol_rel"].iloc[t])
    assert not np.isclose(make_labels(near, h)["lvol_rel"].iloc[t], y["lvol_rel"].iloc[t])
    assert y["valid"].iloc[-h:].sum() == 0


def test_assemble_fills_holes_and_funding_is_not_early():
    df = synthetic(600)
    k = df.reset_index()
    k["open_time"] = (k["time"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    k = k.drop(index=range(100, 105))  # a 5-minute outage
    k["close_time"], k["taker_buy_quote_volume"], k["ignore"] = 0, 0, 0
    p = k[KLINE_COLS].copy()
    p["close"] = df["premium"].to_numpy()[k.index]
    t0 = int(k["open_time"].iloc[0])
    # settles 30s into bar 300: bar 300 closes after it, bar 299 closes before
    f = pd.DataFrame({"t": [t0, t0 + 300 * 60_000 + 30_000], "funding": [1.0, 2.0]})
    out = assemble(k[KLINE_COLS], p, f)
    assert len(out) == 600 and out["filled"].sum() == 5
    assert out["volume"].iloc[100:105].eq(0).all()
    assert out["funding"].iloc[299] == 1.0 and out["funding"].iloc[300] == 2.0


def test_direction_roundtrip_and_independent_calls():
    from forecast.direction import DirectionModel
    from forecast.evaluate import decisions
    df = synthetic(WARMUP + 2000)
    X = build_features(df).iloc[WARMUP:]
    y = make_labels(df, 15).iloc[WARMUP:]
    ok = y["valid"].to_numpy()
    m = DirectionModel.fit(X[ok], y["ret_z"][ok])
    m2 = DirectionModel.from_json(m.to_json())
    assert np.array_equal(m.call(X), m2.call(X))
    assert 0.1 < np.mean(m.call(X[ok]) != 0) < 0.3  # calls only on the extreme ~20%
    k = decisions(np.array([0, 1, 1, -1, 0, 0, 1, 1, 0, -1]), 3)
    assert k.tolist() == [1, 6, 9]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
