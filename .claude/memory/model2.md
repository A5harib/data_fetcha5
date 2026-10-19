---
name: model2
description: Model 2, the self-learning BTC and gold paper bots at /model2 on the HF Space (built 2026-10-19), how they work, backtest, files, ops
metadata:
  type: project
---

The user asked for this on 2026-10-19. They disliked model 1 live: BTC was right 18/44 (41%) and at $95.48, gold 19/35 (54%) and at $96.74. They wanted a separate model that does RL on itself while running on the Space, shown at `/model2`, without touching model 1 or the other instance's 3 new coin models.

## How it works (`backend/model2/`)
- **Data:** Binance **spot** history (gold = PAXGUSDT), the same source the Space sees live. Downloaded by `model2/data.py` to `data/{SYM}_spot_1m_hist.pkl`.
- **Inputs:** model 1's features minus premium and funding, plus 30-day trend and vol regime. That is 38 inputs, and it needs 43,200 bars of warmup.
- **Learner:** recursive least squares on the *sign* of the next-60-min move, with a 90-day forgetting half-life. It updates every hour from minutes whose outcome is known.
- **Styles:** four virtual traders enter on the top 5% / 1% / 0.5% / 0.1% of the last 30 days' scores. Each holds while the score agrees (30 min to 6 h).
- **Self-tuning:** the $100 paper account follows the style with the best 30-day net-of-fee P&L, and **sits out if none is positive** (`MODEL2_SIT_OUT=0` makes it always trade).
- **Fees:** 0.06% a side, all-in, the same as model 1.
- **Shared loop:** one function, `advance()`, drives both the history replay (`model2.train`) and the live loop. `test_model2.py` checks that the live run, with a restart, matches the replay.

## Backtest (prequential, 2022-06 .. 2026-10)
- **BTC:** right 56-72% per year, but it lost money after fees every year ($72, $96, $81, $97, $89 per $100). Without fees it was up in 2023, 2025 and 2026.
- **Gold:** right 60-85%, about break-even after fees ($101, $98, $97, $99, $95). Part of gold's hit rate is likely bid/ask bounce on thin PAXG; model 1's fills have the same issue.
- **Dead ends tried on the way:** a policy-gradient learner (coin flip), a numpy MLP (no better than RLS), 4h and 1d holds (worse), and an "expected profit > fee" entry rule (almost never trades).

## Live (since 2026-10-19 16:45 UTC)
- **Where:** on the Space, started from `app.py` via `model2.attach(fastapi_app)` under the same `PAPER_BOT` switch. Seeds come from `models/model2/{SYM}.json`.
- **State:** saved to dataset `A5harib/paperbot-state` under `model2/{SYM}_agent.json`, after every trade and every 30 min. A restart replays up to about 2 days of missed minutes.
- **First status:** both bots were *sitting out* at launch, because all styles had lost after fees over the prior 30 days. The page also shows each style's live calls, so hit rate is visible even while the account waits.
- **Endpoints:** `/model2` (HTML), `/api/model2`, `/api/model2/{SYM}`.
- **Retrain:** `python -m model2.data --symbol X`, then `python -m model2.train --symbol X`. Upload `models/model2/*` and delete `model2/{SYM}_agent.json` from the dataset to restart fresh.

Related: [[paper-bots]], [[research-findings]], [[gotchas]]
