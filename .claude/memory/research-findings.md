---
name: research-findings
description: What was tested for predicting BTC/XAU up-or-down moves (2026-10-15), what held out of sample, what failed, and why
metadata:
  type: project
---

Four research agents and my own tests ran on 2022-01 to 2026-10 Binance futures 1m data. All selection (features, signs, thresholds) was done on training data only, and the results are out of sample.

## The one effect that holds: snap-back after a stretch
- **What it is:** when price is far from its 4-hour VWAP, near the edge of its 1h, 4h or 1d range, or after a big 60m or 240m run, the next hour usually pulls back a little.
- **Sign:** the sign learned on training data was "fade it" in every quarter, and it held from 2022 to 2026, though weaker in 2025.
- **How it was found:** the best single feature, -vwapdist_240, had a rank correlation of about -0.05 with the next hour's move in every year.
- **Why trees missed it:** the effect lives in the *median* move, not the mean. Ridge or trees on squared loss miss it. A rank-based score, or ridge on a rank-transformed target, finds it.
- **Hit rates:** 56-58% on calls made 10-20% of the time, at 15m, 60m and 240m horizons.
- **The big flaw:** the misses are larger than the hits. It is right on small wiggles and wrong when a real breakout happens (only 41.5% right on the biggest 10% of moves). Expected profit is about zero, and negative after any fees.
- **Independent check:** the "market states" study fixed 20 rules in advance (120 tests, Bonferroni and BH corrected). The survivors were all fade rules:
  - trend extension: 58.2% right, 15/16 quarters, n=2085
  - 15m volume shock: 55.8%
  - 60m shock: 55.1%
  - premium far below its 1-day mean, go long: about 52%
  - The rules replicated on ETH and SOL, which they never saw. None beat a 4bp maker fee.

## Tried, did not fix the win/loss imbalance
- **Skipping calls when the vol model expects a big move:** the hit rate stayed at 55%, and wins and losses both shrank by about the same amount.
- **A safety exit (stop) at 0.5, 0.75, 1.0 or 1.5x the expected move:** cuts losses but turns winners into losers. Profit per call stayed about zero.
- **Requiring buying or selling pressure to already be turning:** fewer calls, the same or worse results.

## Did not work at all
- **XGBoost on ret_z:** right 50.5%, so a coin flip.
- **1-day and 3-day horizons:** nothing beat "always up". There are too few independent calls, a CI of about ±8pp at 1 day and ±13pp at 3 days, so these cannot be tested properly. Always-up wins at 3 days (53.5%) because BTC rose.
- **Funding extremes, funding-settlement minutes, day-of-week and Monday open:** all about 49-51%.
- **Gold:** nothing survived. With 10 months of data, there are too few samples.

## Unfinished when paused
- **Cross-asset features** (ETH and SOL lead or lag, relative strength, market-wide flow) for BTC direction: the agent was stopped before reporting. ETHUSDT and SOLUSDT histories are cached in backend/data.
- **Positioning data** (open interest and long/short ratios from the data.binance.vision daily `metrics` files, 5-min resolution): the agent was stopped while downloading. The live REST equivalents only go back about 30 days.
- **Strength tiers** (are stronger snap-back signals right more often?): the script was stopped. One agent saw 57.6% on the top 10% versus 56.6% on the top 20%, so expect only a small gain.

## Near-misses worth tracking forward
- **BTC weekend gap, continue the Fri 17:00 to Sun 18:00 NY move after the CME reopen:** 59.4% on n=197, 13/16 quarters, and +16bp per call (the only state whose average beat taker cost). It replicated on ETH (60.7%) and SOL (58.2%), but failed multiple-testing correction, and 2024 was 51.9%.

## Earlier pipeline issues worth remembering
- The old `predict/` label was divided by 1-second vol, so 64% of labels hit the ±10 clip.
- Its `seconds_since_trade` feature measured PC clock skew.
- It had only 88 minutes of data.
- At a 30-second horizon the median move (2.7bp) is far below fees (10bp). Even a perfect forecast loses at that horizon.
