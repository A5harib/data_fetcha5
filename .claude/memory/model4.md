---
name: model4
description: Model 4, Kronos forecast paper bots at /model4 (built 2026-11-02), rule, files, deploy gotcha, first check
metadata:
  type: project
---

- Kronos-small (NeoQuasar/Kronos-small, pretrained, not tuned on our data) forecasts each of 7 coins 4 hours ahead after every hourly close, from the last 400 hourly spot candles, averaging 8 samples.
- Rule: if the forecast move is at least 0.5%, bet that coin's whole $100 long or short for 4 hours. 0.1% fee each way, no-fee balance tracked too. Files in `backend/model4/`, Kronos code vendored in `model4/kronos/` (MIT).
- Live on the Space since 2026-11-02 18:54 UTC. Page `/model4`, API `/api/model4`, state `model4/account.json` in PAPER_DATASET. A forecast takes 40-70s on 2 CPU threads.
- **ZeroGPU only accepts torch 2.8.0-2.13.0.** torch 2.5.1 put the whole Space in CONFIG_ERROR (all bots down) until pinned to 2.9.1.
- 20-day check (`python -m model4.check`, 40 steps x 7 coins): it forecast "up" 68% of the time while prices rose only 46% of the time. 68 trades, 38 right (56%), +0.12% per trade after fees. Too small to trust.
- Not committed to git yet.

Related: [[paper-bots]], [[model3]]
