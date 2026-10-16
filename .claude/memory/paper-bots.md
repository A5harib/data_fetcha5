---
name: paper-bots
description: The live fake-money ($100) BTC and gold paper-trading bots on the HF Space, how they are set up, how to check and restart them, and the 6-hourly checker routine
metadata:
  type: project
---

## What they do
`backend/forecast/paper.py`:
- Starts with a fake $100.
- On each snap-back call, puts the whole balance long or short for 60 minutes, then closes.
- Charges 0.06% fee per side, and also tracks a no-fee balance.
- Ignores calls while a trade is open.
- Uses no leverage, so a typical trade moves the balance by ±$0.30-0.40.

The user asked why it bets everything. All-in makes the result read directly as the calls' value. Offering a $10-per-trade mode was left open.

## Where they run (since 2026-10-16 18:34 UTC)
- **The Space:** they run inside the HF Space https://a5harib-a5stocks-backend.hf.space as two daemon threads started by `backend/app.py`, 24/7 whether or not the user's PC is on.
- **Prices:** Binance futures REST is blocked from HF (US), so they use Binance **spot** data (`PAPER_SOURCE=spot`). Over 2026-04..09:
  - BTC on spot: 54.4% right vs 55.1% on futures, with 89% of calls identical.
  - Gold on PAXG: 52.9% vs 54.9%, with only 48% identical. It is a weaker, different bot.
- **The runs:**
  - BTC: 2026-10-16 15:53 to 2026-10-30 15:53 UTC. It started on the PC and continued on the Space, at $100.25 after 2 trades at the switch.
  - Gold (PAXG): a fresh $100 from 2026-10-16 18:34 UTC for 14 days.
  - The earlier gold futures run (PC, 1 trade) is archived in the dataset under `archive/`.
- **Space settings:** variables `PAPER_BOT=1`, `PAPER_SOURCE=spot`, `PAPER_DATASET=A5harib/paperbot-state`, and secret `HF_TOKEN` (the user's write token).
- **Persistence:** state is pulled from the private dataset `A5harib/paperbot-state` on start and pushed after every trade and every 30 min, so a Space restart loses nothing.
- **Viewing:** `/paper?symbol=BTCUSDT|XAUUSDT` is an HTML page refreshed every minute that shows the current UTC and Pakistan time. `/api/paper/{SYM}` returns JSON.
- **Viewer-only mode:** `PAPER_VIEW=1` serves only the dataset's copy. Never set it together with `PAPER_BOT`, because its pull overwrites the live bot's trade file.

## Checking and fixing
- **Stalled:** `last_bar` should be under 2 minutes old. If it is stale, restart the Space at https://huggingface.co/spaces/A5harib/a5stocks-backend. The balance resumes from the dataset.
- **Logs:** `GET https://huggingface.co/api/spaces/A5harib/a5stocks-backend/logs/run` with the HF token (print with `PYTHONIOENCODING=utf-8`).
- **Uploading code changes:** `HfApi().upload_folder(repo_id="A5harib/a5stocks-backend", repo_type="space", folder_path="backend", allow_patterns=[...])`. The Space root equals `backend/`, and each upload restarts the Space.
- **Running on a PC instead:** `PAPER_DATASET=A5harib/paperbot-state python -m forecast.paper --symbol BTCUSDT`, which uses futures data. Stop the Space bots first, so the two don't both trade.

## 6-hourly checker
- **Routine:** "Paper bot check (BTC + gold)", `trig_01NmrwPyF8q8XWTUaTa5ugLq`, https://claude.ai/code/routines/trig_01NmrwPyF8q8XWTUaTa5ugLq
- **Schedule:** cron `28 */6 * * *` UTC.
- **Behaviour:** report-only, never edits or commits. It warns if a bot is stale over 15 min, if the Space is down, or if after 30+ trades a bot is under 45% right or below $95. Warnings also go out as a phone push notification.
- **Open blocker at 2026-10-16:** the cloud sandbox's egress proxy refused `a5harib-a5stocks-backend.hf.space` (403 CONNECT). The user must add that domain to the routine environment's allowed network domains, then re-run the routine to test.
- **Why it doesn't edit the model:** I chose report-only. Retuning in the middle of the test would spoil it and chase noise. The user agreed by saying "yes" to that setup.

## Expectations told to the user
- About 55% of BTC calls right, 53% of gold calls right.
- The balance probably drifts slightly down after fees.
- The no-fee balance shows whether the calls carry value.
- Judge after about 200 trades, i.e. the full 14 days.
