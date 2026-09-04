"""
Feature snapshots for short-horizon price prediction.

One snapshot per second, built only from data available at that instant.
No rolling/shift chains: every field is an explicit lookup, so future data
cannot leak backwards into a feature.

Labeling lives in label_snapshots() and is deliberately in the same file so
the two halves stay readable side by side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

# Feature order is fixed. Model artifacts store this list and refuse to load
# against a mismatched one, so a reordering can never silently mis-feed the model.
FEATURE_NAMES: List[str] = [
    "book_imbalance",
    "book_imbalance_delta_5s",
    "spread_bps",
    "flow_5s",
    "flow_30s",
    "flow_120s",
    "trade_count_5s",
    "trade_count_30s",
    "avg_trade_size_ratio",
    "realized_vol_60s",
    "ret_5s",
    "ret_30s",
    "ret_120s",
    "range_position_120s",
    "seconds_since_trade",
]

# Floor for the volatility divisor. Below this we treat the market as
# effectively still; dividing by a near-zero vol would manufacture huge
# feature values out of rounding noise.
MIN_VOL = 1e-6


@dataclass
class Snapshot:
    """One second of market state, plus the label once it is resolved."""
    t_ms: int
    mid: float
    features: Dict[str, float]
    # Filled in later by label_snapshots(). None means unresolved.
    label: Optional[float] = None
    future_mid: Optional[float] = None
    future_ret: Optional[float] = None

    def to_row(self) -> dict:
        row = {"t_ms": self.t_ms, "mid": self.mid}
        row.update(self.features)
        row["label"] = self.label
        row["future_mid"] = self.future_mid
        row["future_ret"] = self.future_ret
        return row


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not math.isfinite(den):
        return default
    out = num / den
    return out if math.isfinite(out) else default


def build_snapshot(
    t_ms: int,
    mid: float,
    bid_qty: float,
    ask_qty: float,
    best_bid: float,
    best_ask: float,
    mid_history: Sequence[tuple],
    trade_history: Sequence[tuple],
    imbalance_history: Sequence[tuple],
    last_trade_ms: int,
) -> Optional[Snapshot]:
    """
    Assemble one snapshot from trailing state.

    mid_history:       [(t_ms, mid), ...] ascending, covering >= 120s back
    trade_history:     [(t_ms, signed_qty, abs_qty), ...] ascending, >= 120s back
    imbalance_history: [(t_ms, imbalance), ...] ascending, >= 5s back

    Returns None when there is not enough history to fill every field
    honestly. Never returns a snapshot with invented defaults, because a
    fabricated zero looks to the model exactly like a real measured zero.
    """
    if mid <= 0 or not math.isfinite(mid):
        return None
    if not mid_history or not trade_history:
        return None

    # Require a genuine 120s of price history behind us.
    if t_ms - mid_history[0][0] < 120_000:
        return None

    def mid_at(ago_ms: int) -> Optional[float]:
        """Most recent mid at or before t_ms - ago_ms. No interpolation."""
        cutoff = t_ms - ago_ms
        found = None
        for ts, m in reversed(mid_history):
            if ts <= cutoff:
                found = m
                break
        return found

    mid_5s = mid_at(5_000)
    mid_30s = mid_at(30_000)
    mid_120s = mid_at(120_000)
    if mid_5s is None or mid_30s is None or mid_120s is None:
        return None

    def flow_since(ago_ms: int) -> tuple:
        """(signed volume, trade count, absolute volume) over the trailing window."""
        cutoff = t_ms - ago_ms
        signed = 0.0
        absolute = 0.0
        count = 0
        for ts, sq, aq in reversed(trade_history):
            if ts < cutoff:
                break
            signed += sq
            absolute += aq
            count += 1
        return signed, count, absolute

    flow_5s, n_5s, abs_5s = flow_since(5_000)
    flow_30s, n_30s, _ = flow_since(30_000)
    flow_120s, n_120s, abs_120s = flow_since(120_000)

    # Realized volatility: stdev of 1s log returns over the last 60s,
    # the scale everything else gets divided by.
    cutoff_60 = t_ms - 60_000
    recent_mids = [m for ts, m in mid_history if ts >= cutoff_60 and m > 0]
    if len(recent_mids) >= 10:
        logs = np.diff(np.log(np.asarray(recent_mids, dtype=np.float64)))
        realized_vol = float(np.std(logs)) if logs.size else 0.0
    else:
        return None
    vol_scale = max(realized_vol, MIN_VOL)

    # Book imbalance, now and 5s ago.
    total_book = bid_qty + ask_qty
    imbalance = _safe_div(bid_qty - ask_qty, total_book)
    imb_5s_ago = None
    cutoff_imb = t_ms - 5_000
    for ts, im in reversed(imbalance_history):
        if ts <= cutoff_imb:
            imb_5s_ago = im
            break
    if imb_5s_ago is None:
        return None

    # Average trade size, short window against long. > 1 means prints are
    # getting bigger than usual, which is the part worth knowing.
    avg_short = _safe_div(abs_5s, n_5s)
    avg_long = _safe_div(abs_120s, n_120s)
    avg_ratio = _safe_div(avg_short, avg_long, default=1.0)

    # Where in the last 120s range does price currently sit.
    window_mids = [m for ts, m in mid_history if ts >= t_ms - 120_000]
    lo, hi = min(window_mids), max(window_mids)
    range_pos = _safe_div(mid - lo, hi - lo, default=0.5)

    spread_bps = _safe_div(best_ask - best_bid, mid) * 10_000.0

    # Returns are divided by realized vol so a 0.05% move in a calm market and
    # the same move in a violent one are not presented as the same event.
    features = {
        "book_imbalance": imbalance,
        "book_imbalance_delta_5s": imbalance - imb_5s_ago,
        "spread_bps": spread_bps,
        "flow_5s": _safe_div(flow_5s, abs_5s),
        "flow_30s": _safe_div(flow_30s, max(abs_120s, 1e-9)),
        "flow_120s": _safe_div(flow_120s, max(abs_120s, 1e-9)),
        "trade_count_5s": float(n_5s),
        "trade_count_30s": float(n_30s),
        "avg_trade_size_ratio": avg_ratio,
        "realized_vol_60s": realized_vol,
        "ret_5s": _safe_div(math.log(mid / mid_5s), vol_scale),
        "ret_30s": _safe_div(math.log(mid / mid_30s), vol_scale),
        "ret_120s": _safe_div(math.log(mid / mid_120s), vol_scale),
        "range_position_120s": range_pos,
        "seconds_since_trade": (t_ms - last_trade_ms) / 1000.0,
    }

    for name in FEATURE_NAMES:
        v = features.get(name)
        if v is None or not math.isfinite(v):
            return None

    return Snapshot(t_ms=t_ms, mid=mid, features=features)


def label_snapshots(
    snapshots: List[Snapshot],
    horizon_sec: int = 30,
    max_gap_sec: int = 5,
) -> List[Snapshot]:
    """
    Attach the forward return to each snapshot.

    label = log(mid[t + horizon] / mid[t]) / realized_vol_60s[t]

    Vol-normalized, so the model learns direction rather than "was this a
    busy minute". Denominator is the vol measured AT t, which is a feature,
    not future information.

    A snapshot is labeled only if a real observation exists within max_gap_sec
    of the target time. Snapshots near the end of the recording, or across a
    data gap, stay unlabeled and are dropped by callers. This is the only
    place the future is read, which is what keeps leakage auditable.
    """
    if not snapshots:
        return []

    horizon_ms = horizon_sec * 1000
    max_gap_ms = max_gap_sec * 1000
    times = np.array([s.t_ms for s in snapshots], dtype=np.int64)

    for i, snap in enumerate(snapshots):
        target = snap.t_ms + horizon_ms
        j = int(np.searchsorted(times, target, side="left"))
        if j >= len(snapshots):
            continue
        # Nearest observation to the target on either side.
        best = j
        if j > 0 and abs(times[j - 1] - target) < abs(times[j] - target):
            best = j - 1
        if abs(int(times[best]) - target) > max_gap_ms:
            continue
        if best <= i:
            continue

        future_mid = snapshots[best].mid
        if future_mid <= 0 or snap.mid <= 0:
            continue

        raw_ret = math.log(future_mid / snap.mid)
        vol = max(snap.features["realized_vol_60s"], MIN_VOL)
        label = raw_ret / vol
        if not math.isfinite(label):
            continue

        snap.future_mid = future_mid
        snap.future_ret = raw_ret
        snap.label = float(np.clip(label, -10.0, 10.0))

    return snapshots


def snapshots_to_arrays(snapshots: Sequence[Snapshot]):
    """Labeled snapshots -> (X, y, timestamps, raw returns). Unlabeled are dropped."""
    rows, ys, ts, rets = [], [], [], []
    for s in snapshots:
        if s.label is None:
            continue
        rows.append([s.features[n] for n in FEATURE_NAMES])
        ys.append(s.label)
        ts.append(s.t_ms)
        rets.append(s.future_ret)
    if not rows:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ts, dtype=np.int64),
        np.asarray(rets, dtype=np.float64),
    )
