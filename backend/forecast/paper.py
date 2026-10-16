"""
Paper trading: run the saved forecaster on live data with fake money.

Every closed 1m bar: get the live forecast. If flat and the snap-back rule
makes a call, put the whole balance in that direction for `horizon` minutes,
then close. New calls are ignored while a trade is open, the same rule the
walk-forward test scored. Fills are at the close of the signal bar and the
close `horizon` bars later, minus taker fees, so results are slightly
optimistic about fill price and honest about cost.

    python -m forecast.paper --days 14            # start (or resume) a run
    python -m forecast.paper --status             # print the scoreboard

State lives in data/paper/{SYM}_state.json, so a restart resumes the run.
On a Hugging Face Space, whose disk is wiped on restart, set PAPER_DATASET
(a private dataset repo id) and HF_TOKEN: state is pulled from it on start and
pushed to it after every trade.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from forecast.data import LiveFeed
from forecast.predictor import Forecaster

DIR = Path(__file__).resolve().parent.parent / "data" / "paper"
FEE = 0.0006  # per side: Binance taker 0.05% plus a little slippage
log = logging.getLogger("forecast.paper")


def _paths(sym):
    return DIR / f"{sym}_state.json", DIR / f"{sym}_trades.jsonl"


def _save(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def _pull(sym):
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return
    from huggingface_hub import hf_hub_download
    for p in _paths(sym):
        try:
            hf_hub_download(repo, p.name, repo_type="dataset", local_dir=DIR)
            log.info(f"restored {p.name} from {repo}")
        except Exception as e:  # first run: nothing stored yet
            log.info(f"no {p.name} in {repo} yet ({type(e).__name__})")


def _push(sym):
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return
    from huggingface_hub import HfApi
    try:
        HfApi().upload_folder(repo_id=repo, repo_type="dataset", folder_path=DIR,
                              allow_patterns=[p.name for p in _paths(sym)], commit_message="paper trade update")
    except Exception as e:  # never let a storage hiccup stop trading; next trade retries
        log.warning(f"could not save to {repo}: {e}")


def run(sym: str, horizon: int, days: float, start_cash: float):
    DIR.mkdir(parents=True, exist_ok=True)
    _pull(sym)
    state_p, trades_p = _paths(sym)
    if state_p.exists():
        st = json.loads(state_p.read_text())
        log.info(f"resuming run started {st['started']}, ends {st['ends']}")
    else:
        now = datetime.now(timezone.utc)
        st = {"symbol": sym, "horizon": horizon, "started": now.isoformat(),
              "ends": (now + timedelta(days=days)).isoformat(), "start_cash": start_cash,
              "cash": start_cash, "cash_no_fees": start_cash, "position": None, "last_bar": None}
        _save(state_p, st)
    fc = Forecaster(sym, horizon)
    # PAPER_SOURCE=spot on hosts where Binance futures is blocked (US). Prices
    # then come from Binance spot (gold via PAXG), see data.LiveFeed.
    feed = LiveFeed(sym, source=os.environ.get("PAPER_SOURCE", "futures"))
    ends = datetime.fromisoformat(st["ends"])
    pushed = time.time()

    while datetime.now(timezone.utc) < ends or st["position"]:
        try:
            df = feed.get()
            bar = df.index[-1]
            if st["last_bar"] != bar.isoformat():
                st["last_bar"] = bar.isoformat()
                changed = False
                price = float(df["close"].iloc[-1])
                pos = st["position"]
                if pos and bar >= datetime.fromisoformat(pos["exit_bar"]):
                    gross = pos["side"] * (price / pos["entry"] - 1)
                    st["cash"] *= (1 - FEE) * (1 + gross) * (1 - FEE)
                    st["cash_no_fees"] *= 1 + gross
                    t = {**pos, "exit": price, "exit_time": bar.isoformat(),
                         "move_pct": round(100 * gross, 4), "right": gross > 0,
                         "cash": round(st["cash"], 4), "cash_no_fees": round(st["cash_no_fees"], 4)}
                    with trades_p.open("a") as fh:
                        fh.write(json.dumps(t) + "\n")
                    log.info(f"closed {'LONG' if pos['side'] > 0 else 'SHORT'} {pos['entry']} -> {price} "
                             f"({100 * gross:+.3f}%), balance ${st['cash']:.2f}")
                    st["position"] = None
                    changed = True
                if st["position"] is None and datetime.now(timezone.utc) < ends:
                    call = (fc.forecast(df).get("direction") or {}).get("call")
                    if call:
                        side = 1 if call == "UP" else -1
                        st["position"] = {"side": side, "entry": price, "entry_time": bar.isoformat(),
                                          "exit_bar": (bar + timedelta(minutes=horizon)).isoformat()}
                        log.info(f"opened {call} at {price}")
                        changed = True
                _save(state_p, st)
                # Trades are saved at once; otherwise a heartbeat every 30 min,
                # so anyone reading the dataset can tell the bot is alive.
                if changed or time.time() - pushed > 1800:
                    _push(sym)
                    pushed = time.time()
        except Exception as e:  # network blips: log and try again next minute
            log.warning(f"tick failed: {e}")
        # wake a few seconds after the next minute closes
        time.sleep(65 - time.time() % 60)
    log.info("run finished")


_viewed: dict[str, float] = {}


def load(sym: str) -> dict | None:
    # PAPER_VIEW: this process only displays a run that a bot elsewhere writes
    # to PAPER_DATASET (the Space cannot reach Binance). Refresh at most once a minute.
    if os.environ.get("PAPER_VIEW") and time.time() - _viewed.get(sym, 0) > 60:
        DIR.mkdir(parents=True, exist_ok=True)
        _pull(sym)
        _viewed[sym] = time.time()
    state_p, trades_p = _paths(sym)
    if not state_p.exists():
        return None
    st = json.loads(state_p.read_text())
    st["trades"] = [json.loads(l) for l in trades_p.read_text().splitlines()] if trades_p.exists() else []
    return st


def status(sym: str):
    st = load(sym)
    if st is None:
        print("No paper run yet.")
        return
    tr = st["trades"]
    right = sum(t["right"] for t in tr)
    print(f"{sym} paper run {st['started'][:16]} -> {st['ends'][:16]} UTC")
    print(f"balance ${st['cash']:.2f} (started ${st['start_cash']:.2f}); without fees ${st['cash_no_fees']:.2f}")
    if tr:
        print(f"trades {len(tr)}, right {right} ({100 * right / len(tr):.0f}%)")
    pos = st["position"]
    if pos:
        print(f"open: {'LONG' if pos['side'] > 0 else 'SHORT'} at {pos['entry']} since {pos['entry_time'][:16]}, "
              f"closes {pos['exit_bar'][:16]}")
    print(f"last bar seen {st['last_bar']}")


router = APIRouter()


@router.get("/api/paper/{symbol}")
def paper_json(symbol: str):
    return load(symbol.upper()) or {"error": "no paper run"}


@router.get("/paper", response_class=HTMLResponse)
def paper_page(symbol: str = "BTCUSDT"):
    st = load(symbol.upper())
    if st is None:
        return f"<p>No paper run for {symbol.upper()} yet. It starts when the Space boots.</p>"
    tr = st["trades"]
    right = sum(t["right"] for t in tr)
    pos = st["position"]
    rows = "".join(
        f"<tr><td>{t['entry_time'][5:16].replace('T', ' ')}</td><td>{'Long' if t['side'] > 0 else 'Short'}</td>"
        f"<td>{t['entry']:,.1f}</td><td>{t['exit']:,.1f}</td>"
        f"<td class={'up' if t['right'] else 'dn'}>{t['move_pct']:+.2f}%</td><td>${t['cash']:.2f}</td></tr>"
        for t in reversed(tr))
    now = datetime.now(timezone.utc)
    open_line = (f"<p>Open now: <b>{'Long' if pos['side'] > 0 else 'Short'}</b> at {pos['entry']:,.1f}, "
                 f"closes {pos['exit_bar'][11:16]} UTC</p>" if pos else "<p>No open trade. Waiting for a call.</p>")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=60>
<title>Paper bot</title><style>
body{{font:15px system-ui,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;background:#12161f;color:#eaecef}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:6px;border-bottom:1px solid #2b313a;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.up{{color:#0ecb81}}.dn{{color:#f6465d}}.big{{font-size:28px;font-weight:600}}
small{{color:#848e9c}}</style></head><body>
<p><a href="?symbol=BTCUSDT" style="color:#fcd535">BTC</a> &middot; <a href="?symbol=XAUUSDT" style="color:#fcd535">Gold</a></p>
<h2>{symbol.upper()} paper bot</h2>
<div class=big>${st['cash']:.2f}</div>
<small>started ${st['start_cash']:.2f} on {st['started'][:16].replace('T', ' ')} UTC, ends {st['ends'][:16].replace('T', ' ')} UTC.
Without fees: ${st['cash_no_fees']:.2f}</small>
<p>Trades: {len(tr)}, right: {right}{f" ({100 * right / len(tr):.0f}%)" if tr else ""}</p>{open_line}
<table><tr><th>Opened (UTC)</th><th>Side</th><th>Entry</th><th>Exit</th><th>Move</th><th>Balance</th></tr>{rows}</table>
<p><small>Now: {now:%H:%M} UTC ({now + timedelta(hours=5):%H:%M} Pakistan). Bot last saved at {st['last_bar'][11:16] if st['last_bar'] else '-'} UTC;
it saves on every trade and every 30 minutes, so up to 30 minutes old is normal. Page refreshes every minute.</small></p>
</body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--cash", type=float, default=100.0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status(a.symbol.upper())
    else:
        DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                            handlers=[logging.FileHandler(DIR / f"{a.symbol.upper()}_paper.log"), logging.StreamHandler()])
        run(a.symbol.upper(), a.horizon, a.days, a.cash)
