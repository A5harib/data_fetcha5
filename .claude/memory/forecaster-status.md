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
- **XAUUSDT 60m:** trained and saved 2026-10-16. The range model beats HAR in 5/7 months. Up/down is right 53.3% vs 50.3% always-up, 5/7 months, and only just passes the gate.
- **Committed** by the user as `0c50290 mid-push`, except `backend/models/forecast/XAUUSDT_60m/` (added 2026-10-16, uncommitted).
- **Cached data:** `backend/data/*_1m_hist.pkl` and `backend/data/raw/` are gitignored. ETHUSDT and SOLUSDT histories were downloaded too, for the cross-asset research.
- **Research scripts** were in the session's temp scratchpad and are gone. [[research-findings]] has the results.

## Paper-trading run (started 2026-10-16 15:53 UTC, ends 2026-10-30 15:53 UTC)
`forecast/paper.py`: fake $100 on BTCUSDT, all-in long/short on each snap-back call, hold 60 min, 0.06% fee per side, ignores calls while a trade is open. Started as a hidden background process. Check with `python -m forecast.paper --status` (from backend/). Files in `backend/data/paper/` (gitignored): state, trades.jsonl, paper.log. If the PC restarts, rerun `python -m forecast.paper` to resume. First trade: LONG at 81,403.6.
Expectation set with the user: about 55% right, balance likely slightly down after fees, the "without fees" balance shows whether the calls carry value.
Cloud move (prepared 2026-10-16): user chose Oracle Cloud Always Free. Kit in `deploy/oracle/` (README.md guide for the user, setup.sh, paperbot.service, requirements-paper.txt, pack.py builds paperbot.zip with the live state). Region must be non-US (Binance blocks US); guide says Frankfurt. paper.py now uses `data.LiveFeed` (full 4000-bar fetch once, then the last 5 bars per minute, verified identical to fetch_recent). The PC bot (old code) keeps running until the user does step 4 of the guide.

## Where the bots run now (2026-10-16 22:20 PKT)
- Oracle signup failed for the user. The HF Space cannot run the bots: Binance REST returns HTTP 451 from HF (US region), even though the websocket gets through.
- Bots run on the user's PC: two hidden `python -m forecast.paper --symbol BTCUSDT|XAUUSDT` processes started with env `PAPER_DATASET=A5harib/paperbot-state`. They push state to that private HF dataset after every trade and every 30 min as a heartbeat. Gold started a fresh $100 at 2026-10-16 17:20 UTC.
- Space https://a5harib-a5stocks-backend.hf.space runs with `PAPER_VIEW=1`, `PAPER_DATASET`, and secret `HF_TOKEN`. `/paper?symbol=BTCUSDT|XAUUSDT` and `/api/paper/{SYM}` show results pulled from the dataset. The Space code was synced with the repo (it had been missing the ml v3 files).
- If the PC sleeps the bots stop; restart both with the same env. The longer-term fix is a non-US VPS (Hetzner).

## Update 2026-10-16 ~18:35 UTC: bots now run ON the Space
- Binance futures REST is blocked from HF (451), but Binance's spot market-data host `data-api.binance.vision` answers. `LiveFeed(source="spot")` uses it (premium=0, funding=0; only the range model uses those). Gold maps to PAXGUSDT.
- Spot check over 2026-04..09 with the saved models: BTC calls 54.4% right on spot vs 55.1% on futures, with 89% of calls identical. Gold via PAXG: 52.9% vs 54.9%, only 48% of calls identical, so it is a weaker, different bot.
- Space variables: `PAPER_BOT=1`, `PAPER_SOURCE=spot`, `PAPER_DATASET=A5harib/paperbot-state`, secret `HF_TOKEN`. PAPER_VIEW is removed; never set it together with PAPER_BOT, because the viewer pull would overwrite the live bot's trade file.
- BTC continued its run ($100.25 after 2 trades). Gold restarted with a fresh $100 on PAXG at 18:34 UTC. The old gold futures run is archived in the dataset under archive/.
- PC bots are stopped.
- 6-hourly checker routine trig_01NmrwPyF8q8XWTUaTa5ugLq (cron 28 */6 UTC, report-only, push notification on warnings). The cloud network blocked hf.space on the first test run; the user must add `a5harib-a5stocks-backend.hf.space` to the environment's allowed domains.
