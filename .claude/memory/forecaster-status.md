---
name: forecaster-status
description: State of the new price forecaster in backend/forecast as of 2026-10-15, how to run it, and what is not committed yet
metadata:
  type: project
---

Built from scratch on 2026-10-15 in `backend/forecast/`. The user called the old `backend/ml/` (reversal model) and `backend/predict/` code trash and asked for a fresh design. Neither old folder is used by the new code, and both are still in the repo.

## What it predicts (60-minute horizon)
1. **Price range.** It gives the band the price lands in 80% of the time, plus a "9 in 10" floor and ceiling. An XGBoost volatility model drives it.
2. **Up/down call.** This is a "snap-back" rule (`direction.py`). It calls direction only when price is stretched far from its recent average and range, and bets on a pullback. It makes a call roughly 20% of the time and otherwise says "no call".

## Files
- `data.py` downloads Binance USD-M futures 1m klines, the 1m premium index and funding.
  - Training data comes from data.binance.vision. Live data comes from REST. One `assemble()` builds both, so they match.
  - Exchange outages and missing premium days are flagged.
- `features.py` holds 35 past-only, scale-free features and the labels. `FUNDING_HOURS` covers the funding interval: XAU settles every 4h, BTC every 8h.
- `evaluate.py` runs walk-forward with an expanding window, 3-month test blocks and an embargo. It compares the vol model with HAR and the up/down calls with "always up", and scores calls without overlap via `decisions()`.
- `direction.py` is the snap-back rule: winsorize, z-score, take the top 5 features by rank IC (chosen on train only), average them, and call only above the train 80th percentile.
- `train.py` evaluates, gates, and saves to `$MODEL_DIR/forecast/{SYM}_{H}m/` (`vol.json`, `dir.json`, `meta.json`).
- `predictor.py` exposes `GET /api/forecast/{symbol}?horizon=60`. It recomputes once per closed bar and is mounted in `main.py`.
- `test_forecast.py` holds the self-checks: no future data in features, exact label windows, gap filling, the funding as-of, and the direction round-trip.
- `src/components/ML/ForecastPanel.jsx` is the UI. It replaced `ReversalPanel` in `App.jsx`, and `ReversalPanel.jsx` is still on disk.
- `app.py` now seeds `models/` recursively, so the HF Space gets `models/forecast/`.

## Commands (run from backend/)
```
python -m forecast.data --symbol BTCUSDT --start 2022-01      # download + assemble history (~2.5M bars)
python -m forecast.test_forecast                               # self-checks, no network
python -m forecast.train --symbol BTCUSDT --horizon 60         # ~15 min on 4 cores: eval, gate, save
python -m forecast.train --symbol XAUUSDT --horizon 60 --min-train-months 3 --test-months 1
python -m forecast.predictor BTCUSDT 60                        # one live forecast
```

## State when paused (2026-10-15)
- **BTCUSDT 60m:** trained and saved, and both parts passed their gates. A live forecast was tested and works.
- **XAUUSDT 60m:** not saved. The training run was stopped mid-way. Rerun the command above, and expect the up/down part to fail its gate (see [[forecaster-results]]).
- **Not committed.** The new folder, the edits to main.py, app.py, App.jsx and backend/.gitignore, and `models/forecast/` are all uncommitted.
- **Cached data:** `backend/data/*_1m_hist.pkl` and `backend/data/raw/` are gitignored. ETHUSDT and SOLUSDT histories were downloaded too, for the cross-asset research.
- **Research scripts** were in the session's temp scratchpad and are gone. [[research-findings]] has the results.
