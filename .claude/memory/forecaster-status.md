---
name: forecaster-status
description: What the price forecaster in backend/forecast is, its files and commands, what is trained, and what is committed (as of 2026-10-16)
metadata:
  type: project
---

Built from scratch on 2026-10-15 in `backend/forecast/`. The user called the old `backend/ml/` (reversal model) and `backend/predict/` code trash and asked for a fresh design. The new code uses neither. Both are still in the repo, and the Space still imports `ml/` through `main.py`.

## What it predicts (60-minute horizon)
1. **Price range.** It gives the band the price lands in 80% of the time, plus a "9 in 10" floor and ceiling. An XGBoost volatility model drives it.
2. **Up/down call.** The "snap-back" rule in `direction.py` calls direction only when price is stretched far from its recent average and range, and bets on a pullback. It makes a call about 20% of the time and otherwise says "no call".

Results are in [[forecaster-results]]. Everything tried for up/down is in [[research-findings]]. The live paper-trading bots are in [[paper-bots]]. Traps hit along the way are in [[gotchas]].

## Files (backend/forecast/)
- `data.py`:
  - Training reads Binance USD-M futures 1m klines, the premium index and funding from data.binance.vision. `assemble()` builds one canonical frame.
  - `fetch_recent()` reads live futures REST. `LiveFeed` fetches 4000 bars once, then the last 5 bars each minute, and gives the same output.
  - `LiveFeed(source="spot")` reads Binance spot (`data-api.binance.vision`) for hosts where futures is blocked. Premium and funding are 0 there, and gold maps to PAXGUSDT.
- `features.py`: 35 past-only, scale-free features and the labels. `FUNDING_HOURS` sets XAU to 4h and the rest to 8h.
- `evaluate.py`:
  - Walk-forward with an expanding window, 3-month test blocks and an embargo.
  - The vol model is compared with HAR, and the up/down calls with "always up".
  - `decisions()` keeps only non-overlapping calls.
- `direction.py`: the snap-back rule. It winsorizes and z-scores, takes the top 5 features by rank IC (on train only), averages them, and calls only above the train 80th percentile.
- `train.py`: evaluates, gates, and saves to `$MODEL_DIR/forecast/{SYM}_{H}m/` (`vol.json`, `dir.json`, `meta.json`). Writes are atomic.
- `predictor.py`: serves `GET /api/forecast/{symbol}?horizon=60`, recomputed once per closed bar.
- `paper.py`: the fake-money bot, plus `GET /paper` and `/api/paper/{SYM}`. See [[paper-bots]].
- `test_forecast.py`: self-checks covering no future data in features, exact label windows, gap filling, the funding as-of, and the direction round-trip.
- UI: `src/components/ML/ForecastPanel.jsx` replaced `ReversalPanel` in `App.jsx`. `ReversalPanel.jsx` is still on disk.

## Commands (run from backend/)
```
python -m forecast.data --symbol BTCUSDT --start 2022-01      # download + assemble history (~2.5M bars)
python -m forecast.test_forecast                               # self-checks, no network
python -m forecast.train --symbol BTCUSDT --horizon 60         # ~15 min on 4 cores: eval, gate, save
python -m forecast.train --symbol XAUUSDT --horizon 60 --min-train-months 3 --test-months 1
python -m forecast.predictor BTCUSDT 60                        # one live forecast
```

## Trained models (both trained through 2026-10-13 22:59 UTC)
- BTCUSDT_60m: both parts passed their gates.
- XAUUSDT_60m: the range part passed. Up/down only just passed (53.3% vs 50.3% always-up, 5/7 months).

## Git state (2026-10-16)
- **Committed** by the user as `0c50290 mid-push`: the forecaster, the BTC model and the UI.
- **Uncommitted:**
  - `forecast/paper.py`
  - the `LiveFeed` and spot changes in `data.py`
  - the bot startup in `app.py` and the router in `main.py`
  - `models/forecast/XAUUSDT_60m/`
  - `deploy/oracle/` (unused, Oracle failed)
  - the gitignore entries
  - these memory notes
- The **Hugging Face Space has the newer code**. It was uploaded directly, so the Space is ahead of GitHub.
- **Cached data (gitignored):** `backend/data/*_1m_hist.pkl` (BTC, XAU, ETH, SOL) and `backend/data/raw/`.
