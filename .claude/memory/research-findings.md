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

## Finished 2026-10-16 (none fixes the money problem)
- Cross-asset (ETH, SOL returns, flow, relative strength) added to or filtering the snap-back score: hit 54.5-55.6% vs 55.6% base, same win/loss shape. No gain.
- Learned veto (xgb on earlier quarters' calls, nested walk-forward): skipping predicted big losses or low predicted payoff lifts avg per call from +0.8 to +1.5bp, 13/14 quarters positive, hit 55.8%. Still under the 4bp maker fee.
- Limit-order entries 0.1-0.5 sigma beyond price, waiting 5-30 min: adverse selection. Fills happen when the move keeps running, avg per call drops to about -0.5bp.
- Positioning (data.binance.vision daily metrics, 5-min OI and long/short ratios, cached in session scratchpad only): univariate ICs all within +/-0.03. As a filter on snap-back calls: "OI fell over last 240m" gives 56.7% right, +1.5bp, 14/16 quarters positive, on half the calls. "Crowd long/short ratio leans with the stretch" gives 56.1%. Small, not shipped; live REST history (~30 days) would be enough to compute it.
- Verdict: the snap-back calls are right about 55-57% of the time but are worth roughly zero money per call. Every fix tried moved profit by at most 1bp against a 4-12bp cost.

## Near-misses worth tracking forward
- **BTC weekend gap, continue the Fri 17:00 to Sun 18:00 NY move after the CME reopen:** 59.4% on n=197, 13/16 quarters, and +16bp per call (the only state whose average beat taker cost). It replicated on ETH (60.7%) and SOL (58.2%), but failed multiple-testing correction, and 2024 was 51.9%.

## Earlier pipeline issues worth remembering
- The old `predict/` label was divided by 1-second vol, so 64% of labels hit the ±10 clip.
- Its `seconds_since_trade` feature measured PC clock skew.
- It had only 88 minutes of data.
- At a 30-second horizon the median move (2.7bp) is far below fees (10bp). Even a perfect forecast loses at that horizon.

## RL trading agents (2026-10-16, numpy CEM/ES policies, long/flat/short)
Trained 2022..2026-06, picked on 2026-07..09, tested once on 2026-10-01..13 and live 10-14..16, 6bp fee per side.
- BTC 60m, trained at taker fees: learned to stay out ~99% of the time (+0.9% test from 1 trade, 0 live trades).
- BTC 60m trained at 2bp and BTC 15m: picked because they rode the Jul-Sep bull run; test/live -5.8% to -4.2%. Long training runs hit +300% to +3800% in-sample and lost on unseen months (memorised).
- Gold 15m: +20% on pick months, -2.7% on the honest test, +3.3% live from 2 trades (noise).
- Verdict: RL adds nothing over the snap-back rule; best learned behaviour is sitting out.
- Live forward test of the saved forecasters (2.6 days): BTC range 79%, 34 calls 62% right; gold range 81%, 40 calls 6 in 10 right. Both still below fees per call. Too few calls to judge.
