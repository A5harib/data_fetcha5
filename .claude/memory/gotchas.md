---
name: gotchas
description: Traps hit while building and deploying the forecaster and bots (Binance geo-blocks, HF Space, cloud routines, Windows, pandas 3) and the fix for each
metadata:
  type: reference
---

## Binance data access
- **Futures REST blocks US servers.** `fapi.binance.com` returns HTTP 451 "restricted location" from US hosts: the HF Space, and likely Google Cloud's free tier, which is US-only. The futures **websocket** (fstream) still got through from HF, so a websocket-only test can falsely show that "Binance works".
- **Spot market data works from US hosts.** `data-api.binance.vision` (`/api/v3/klines`, 1000 bars per page) answers from HF. It is the fallback (`LiveFeed(source="spot")`).
- **Spot archive timestamps changed.** data.binance.vision spot kline files from 2025 on use **microsecond** timestamps. Divide by 1000 when the value is over 1e14.
- **Futures archive coverage:**
  - Daily `bookTicker` stopped after early 2024.
  - `aggTrades`, `bookDepth`, `klines`, `premiumIndexKlines` and daily `metrics` (5-min OI and long/short ratios, from 2022, ratios from 2022-01-30) are current.
  - Futures `metrics` exist only as daily files, not monthly.
- **Two settlement intervals.** XAUUSDT funding settles every **4h**, BTC every 8h. The listing went live 2025-12-11.
- **Finalization lag.** The premium index of the just-closed bar can change a few seconds after the close. Re-reading the last bars each minute fixes it.
- **Flaky connection from the user's PC.** Binance connections from the user's PC (Pakistan) sometimes get reset (WinError 10054, curl exit 35). Retry each minute rather than crash.

## Hugging Face Space
- **Layout.** The Space root equals `backend/`.
- **The Space can lag GitHub.** It once lacked the `ml/` v3 files, and `main.py` then crashed on import (RuntimeError on boot). Sync all code, not just the changed files.
- **Free Spaces sleep** after about 48h with no visitors, and a restart wipes local files. Persist to a private dataset.
- **The `hf` CLI is not on PATH** on the user's Windows machine. Log in with `python -c "from huggingface_hub import login; login()"` in the user's own terminal. Never paste tokens in chat.
- **Reading the Space log needs UTF-8.** It contains progress-bar characters that crash Windows cp1252 printing, so set `PYTHONIOENCODING=utf-8`.

## Cloud routines (claude.ai/code/routines)
- **The sandbox egress proxy blocks unlisted hosts** (403 on CONNECT, e.g. `*.hf.space`). Add domains to the environment's allowed list.
- **`PushNotification` works from a routine** and reaches the user's phone.

## Local tooling
- **pandas 3.0 datetime resolution.** `pd.to_datetime(unit="ms")` gives `datetime64[ms]`, so `.astype("int64")` is already in milliseconds. `merge_asof` refuses to join ms with us keys, so cast with `.astype("datetime64[ms, UTC]")`.
- **Windows PowerShell zips break on Linux.** `Compress-Archive` writes backslash paths that unzip wrongly there. Build zips with Python's `zipfile` and strip CRLF from .sh and .service files.
- **No RL libraries.** torch, gymnasium and stable-baselines3 are not installed. The RL experiments used numpy CEM/ES instead.
- **Only 4 CPU cores.** Never run two full walk-forward trainings at once. Each takes about 15 min on 2.5M rows.
- **Background shell commands.** Don't run `cat >>` with no stdin in a background shell. It hangs waiting for input.

## Signups
- **Oracle Cloud Always Free failed** for the user (2026-10-16, details not given). Don't suggest it again without asking what went wrong.
