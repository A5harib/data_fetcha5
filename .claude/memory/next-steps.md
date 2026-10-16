---
name: next-steps
description: Open items for the forecaster and paper bots, in priority order (updated 2026-10-16)
metadata:
  type: project
---

1. **Unblock the 6-hourly checker.** The user adds `a5harib-a5stocks-backend.hf.space` to the routine environment's allowed domains. Then run routine `trig_01NmrwPyF8q8XWTUaTa5ugLq` and read its log to confirm it works. See [[paper-bots]].
2. **Commit and push the uncommitted work** once the user approves. The list is in [[forecaster-status]]. The Space is ahead of GitHub until then.
3. **Judge the paper runs when they end** (BTC 2026-10-30 15:53 UTC, gold 2026-10-30 ~18:34 UTC):
   - % of calls right versus about 55% (BTC) and 53% (gold).
   - The no-fee balance versus the with-fee balance.
   - Report in plain words.
4. **The money problem is unsolved.** The calls are right more often than wrong, but they are worth about 0 per trade after fees. Every fix tried moved profit by at most about 1.5bp against 4-12bp costs, and that includes RL. See [[research-findings]]. Untested ideas:
   - weekly holding periods with trend features
   - funding-rate carry instead of direction
   - an OI-fell-over-4h filter (56.7% right on half the calls; needs live OI)
5. **Optional:** a $10-per-trade bot mode, or auto-start of the PC bots on Windows boot (only if the bots move back to the PC).
6. **Cleanup offered but never approved:**
   - The old `backend/ml/`, `backend/predict/`, `ReversalPanel.jsx` and `deploy/oracle/` are unused.
   - A Polygon API key leaked in `src/services/polygonService.js`, which is dead code. The key is in git history and needs rotating.
   - The paper-trading SELL bug in App.jsx lets you sell without holding.
   - A line that looked like a password was pasted at the bottom of `deploy/oracle/README.md`. The user was told and asked to delete it and change it if real. Check it's gone before committing.
