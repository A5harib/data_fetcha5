"""Checks the model 4 trading rule without Kronos or the network.

    python -m pytest model4/test_model4.py
"""
from model4.bot import FEE, THRESHOLD, mark, new_book, step


def test_small_forecast_does_nothing():
    b = new_book()
    assert step(b, "2026-11-02T10:00:00+00:00", 100.0, THRESHOLD / 2) == []
    assert b["pos"] is None and b["cash"] == 100.0


def test_long_round_trip_charges_two_fees():
    b = new_book()
    step(b, "2026-11-02T10:00:00+00:00", 100.0, 1.0)
    assert b["pos"]["side"] == 1
    step(b, "2026-11-02T13:00:00+00:00", 150.0, 5.0)        # still open, a new call is ignored
    assert b["pos"]["entry"] == 100.0 and b.get("n", 0) == 0
    step(b, "2026-11-02T14:00:00+00:00", 102.0, None)       # 4 hours later: close
    assert b["pos"] is None and b["n"] == 1 and b["right"] == 1
    assert abs(b["cash"] - 100 * (1 - FEE) ** 2 * 1.02) < 1e-9
    assert abs(b["nofee"] - 102.0) < 1e-9


def test_short_loses_when_price_rises_and_reopens_same_bar():
    b = new_book()
    step(b, "2026-11-02T10:00:00+00:00", 100.0, -1.0)
    assert abs(mark(b, 101.0)[1] - 99.0) < 1e-9
    msgs = step(b, "2026-11-02T14:00:00+00:00", 101.0, 2.0)
    assert len(msgs) == 2 and b["right"] == 0 and b["pos"]["side"] == 1
    assert b["pos"]["cash_in"] < 99.0
