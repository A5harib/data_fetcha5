---
name: next-steps
description: Prioritized to-do list for the forecaster when work resumes (paused 2026-10-15)
metadata:
  type: project
---

In priority order.

1. **Decide with the user what "good" means now.** Accuracy is done (BTC 55.5%, XAU 53.3%, honest). Money is not: every fix tried, including the cross-asset, veto, limit-entry and positioning ones, is worth at most +1.5bp per call against 4-12bp fees (see [[research-findings]]).
2. **Optional small accuracy gain:** filter BTC calls with "open interest fell over the last 4h", which gives 56.7% on half the calls. It needs live OI from REST `/futures/data/openInterestHist`.
3. **Untested ideas if money matters:** much longer holding periods with trend features (weekly), or a different market with higher fees on the other side, e.g. funding-rate carry rather than direction.
4. **Run the snap-back rule live and log its calls.** Check that the real hit rate matches the roughly 55% from testing before anyone relies on it.
5. **Commit `models/forecast/XAUUSDT_60m/`** once the user approves. Also check that the HF Space requirements cover scipy (it comes with scikit-learn).
6. **Cleanup offered earlier but never approved:**
   - The old `backend/ml/` and `backend/predict/`, and `ReversalPanel.jsx`, are now unused by the new UI.
   - There's a leaked Polygon API key in `src/services/polygonService.js`, which is dead code, and the key is in git history. It needs rotating.
   - The paper-trading SELL bug in App.jsx lets you sell without holding.
