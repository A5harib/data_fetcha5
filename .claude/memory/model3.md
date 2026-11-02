---
name: model3
description: Model 3, the weekly trend-rotation paper bot at /model3 (built 2026-10-28), rule, backtest, research behind it, files, status
metadata:
  type: project
---

The user called models 1 and 2 trash on 2026-10-28 because they kept losing money. Live after 9 days, model 1's 7 bots were all down ($88-96) but 6 of 7 were up before fees ($100-108). Fees on about 150 trades were the killer, so model 3 trades about once a week per sleeve.

## Rule (`backend/model3/strategy.py`)
- 7 assets: BTC, ETH, SOL, BNB, XRP, DOGE, gold (PAXG), Binance spot daily closes.
- Trending = above its 50-day average and up over 20 days. Hold the top 2 by 20-day rise, each sized min(1, 0.4 / yearly vol) / 2, rest in cash.
- **7 sleeves**, one per weekday, each rebalancing weekly right after 00:00 UTC. The weekday alone swung the backtest from $292 to $837 per $100, so sleeves average that luck out.
- Fee 0.1% a side. `step()` is shared by the backtest and the live bot; `test_model3.py` checks that they match.

## Backtest (0.1% fees)
- 2022-06..2026-10: $100 to $400, worst drop -26%. By year: 2022 -7%, 2023 +61%, 2024 +95%, 2025 +54%, 2026 YTD -11%.
- Unseen 2025-01..2026-10: $136, worst drop -25%. Holding all 7 gave $82.
- Vol target 0.4 was picked on 2022-24 (0.6 earned more but fell further). A BTC-200-day filter only helped on the test period (hindsight), so it was not adopted. Trailing stops and 3-day rebalancing did not help.
- 2018-2021 check (never seen): the same rule lost 39% in 2018 while BTC lost 78%, and made about 10x over 2018-21. It is crash protection, not a no-loss machine.
- Funding carry (spot long, perp short) was rejected. It pays about 1-2% a year in 2025-26.

## Status
- On 2026-10-28 every asset was falling, so the first live run would sit 100% in cash.
- Wired into `app.py` under PAPER_BOT. State is saved to `model3/account.json` in PAPER_DATASET. Page `/model3`, API `/api/model3`.
- Retrain/refresh the summary: `python -m model3.data && python -m model3.train`.
- Live on the Space since 2026-10-28 15:27 UTC with $100, starting in all cash. All 7 sleeves count 2026-10-27 as their first rebalance, so each buys on its own weekday from then on. Not committed to git yet.

Related: [[paper-bots]], [[model2]], [[research-findings]]
