"""
Model 3 has nothing to fit: the rule is fixed in strategy.py. This runs the
backtest on the downloaded history and saves the summary the /model3 page
shows.

    python -m model3.data && python -m model3.train
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from model3 import strategy as st
from model3.data import load

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models")) / "model3"


def summarize(closes, start: str) -> dict:
    books = st.backtest(closes, start)
    e = st.equity(books)
    hold = closes.loc[start:].iloc[[0, -1]]
    return {"start": start, "end": str(e.index[-1].date()), "per_100": round(float(e.iloc[-1]), 2),
            "worst_drop_pct": round(float(100 * (e / e.cummax() - 1).min()), 1),
            "hold_all_per_100": round(float(100 * (hold.iloc[-1] / hold.iloc[0]).mean()), 2),
            "hold_btc_per_100": round(float(100 * hold["BTCUSDT"].iloc[-1] / hold["BTCUSDT"].iloc[0]), 2),
            "fees_per_100": round(sum(b.fees_paid for b in books), 2),
            "years": {y: {k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()}
                      for y, r in st.yearly(books).items()}}


def main():
    c = load()
    meta = {"rule": {"lookback": st.LOOKBACK, "trend_sma": st.TREND_SMA, "top_k": st.TOP_K,
                     "vol_target": st.VOL_TARGET, "sleeves": st.SLEEVES, "fee": st.FEE},
            "all": summarize(c, "2022-06-01"), "unseen": summarize(c, "2025-01-01")}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
