---
name: next-steps
description: Prioritized to-do list for the forecaster when work resumes (paused 2026-10-15)
metadata:
  type: project
---

In priority order.

1. **Retrain XAUUSDT.** Run `python -m forecast.train --symbol XAUUSDT --horizon 60 --min-train-months 3 --test-months 1`.
   - The range model should ship.
   - Up/down will probably fail its gate on gold. If so, the panel says "not available", which is correct.
   - Without a saved XAU model, the panel shows "No forecaster trained for this symbol" on gold.
2. **Fix the win/loss imbalance in up/down calls.** This is the real blocker to making money, as opposed to being right more often than wrong. Ideas not yet tested:
   - Predict the *size* of the snap-back and only call when the expected win clearly exceeds the expected loss.
   - Detect breakouts with a separate classifier, using shock plus volume plus flow persistence, and veto calls in those conditions.
   - Turn it into a maker-only, limit-order entry at a better price than the market.
3. **Finish the stopped research.** This covers cross-asset features (ETH and SOL are already cached), positioning data (metrics files), and strength tiers. See [[research-findings]].
4. **Run the snap-back rule live and log its calls.** Check that the real hit rate matches the roughly 55% from testing before anyone relies on it.
5. **Commit once the user approves.** The new `backend/forecast/` folder, `models/forecast/BTCUSDT_60m/`, the UI panel, and the edits to main.py, app.py, App.jsx and backend/.gitignore are all uncommitted. Also check that the HF Space requirements cover scipy (it comes with scikit-learn).
6. **Cleanup offered earlier but never approved:**
   - The old `backend/ml/` and `backend/predict/`, and `ReversalPanel.jsx`, are now unused by the new UI.
   - There's a leaked Polygon API key in `src/services/polygonService.js`, which is dead code, and the key is in git history. It needs rotating.
   - The paper-trading SELL bug in App.jsx lets you sell without holding.
