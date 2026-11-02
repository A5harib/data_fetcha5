"""
Model 4 live: Kronos forecasts on hourly Binance spot candles, one fake $100
per coin, shown at /model4.

After each hourly close it asks Kronos (forecaster.py) where each coin will be
4 hours from now. If the forecast move is at least THRESHOLD %, it puts that
coin's whole balance long (forecast up) or short (forecast down), holds for 4
hours, then closes. No leverage. 0.1% fee each way, and a no-fee balance is
tracked beside it so you can see whether the calls themselves carry value.
Shorts are fake-money only: spot can't short, the bot just books the result.

Memory: the account is saved locally and pushed to PAPER_DATASET under model4/
after every trade and every 30 minutes. Other models' files are never touched.

    MODEL4_BOT=1 python -m model4.bot      # run locally
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from model4.data import SYMBOLS

DIR = Path(__file__).resolve().parent.parent / "data" / "model4"
LOCAL = DIR / "account.json"
REMOTE = "model4/account.json"
START_CASH = 100.0
FEE = 0.001          # per side
THRESHOLD = 0.5      # % forecast move needed to trade; round-trip fees are 0.2%
HOLD_HOURS = 4
NAMES = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB",
         "XRPUSDT": "XRP", "DOGEUSDT": "DOGE", "XAUUSDT": "Gold"}
log = logging.getLogger("model4.bot")
_live: dict = {}   # latest prices, for the page


# ---------------------------------------------------------------- trading rule (pure, tested)

def new_book() -> dict:
    return {"cash": START_CASH, "nofee": START_CASH, "pos": None, "trades": [], "right": 0, "last_forecast": None}


def step(book: dict, bar_time: str, price: float, move_pct: float | None) -> list[str]:
    """Apply one hourly close to one coin's book. Returns what happened, for the log."""
    did = []
    pos = book["pos"]
    if pos and bar_time >= pos["close_at"]:
        ret = pos["side"] * (price / pos["entry"] - 1)
        book["cash"] = pos["cash_in"] * (1 - FEE) * (1 + ret) * (1 - FEE)
        book["nofee"] = pos["nofee_in"] * (1 + ret)
        won = ret > 0
        book["right"] += int(won)
        book["trades"].append({**{k: pos[k] for k in ("side", "entry", "opened", "forecast")},
                               "exit": price, "closed": bar_time, "ret_pct": round(100 * ret, 3),
                               "balance": round(book["cash"], 4)})
        book["trades"] = book["trades"][-200:]
        book["n"] = book.get("n", 0) + 1
        book["pos"] = pos = None
        did.append(f"closed {'right' if won else 'wrong'} {100 * ret:+.2f}%")
    if move_pct is not None:
        book["last_forecast"] = {"at": bar_time, "move_pct": round(move_pct, 3)}
        if pos is None and abs(move_pct) >= THRESHOLD:
            close_at = (datetime.fromisoformat(bar_time) + timedelta(hours=HOLD_HOURS)).isoformat()
            book["pos"] = {"side": 1 if move_pct > 0 else -1, "entry": price, "opened": bar_time, "close_at": close_at,
                           "forecast": round(move_pct, 3), "cash_in": book["cash"], "nofee_in": book["nofee"]}
            did.append(f"opened {'long' if move_pct > 0 else 'short'} on forecast {move_pct:+.2f}%")
    return did


def mark(book: dict, price: float | None) -> tuple[float, float]:
    """Balance and no-fee balance at `price`, counting an open trade as if closed now."""
    pos = book["pos"]
    if not pos or price is None:
        return book["cash"], book["nofee"]
    ret = pos["side"] * (price / pos["entry"] - 1)
    return pos["cash_in"] * (1 - FEE) * (1 + ret) * (1 - FEE), pos["nofee_in"] * (1 + ret)


# ---------------------------------------------------------------- storage

def _pull() -> bool:
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return False
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, REMOTE, repo_type="dataset")
        DIR.mkdir(parents=True, exist_ok=True)
        LOCAL.write_text(Path(p).read_text())
        log.info(f"restored model 4 from {repo}")
        return True
    except Exception as e:  # first run: nothing stored yet
        log.info(f"no model 4 state in {repo} yet ({type(e).__name__})")
        return False


def _push():
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    try:
        HfApi().create_commit(repo_id=repo, repo_type="dataset", commit_message="model 4 update",
                              operations=[CommitOperationAdd(REMOTE, str(LOCAL))])
    except Exception as e:  # never let storage stop trading; the next push retries
        log.warning(f"could not save model 4 to {repo}: {e}")


def _save(acct: dict):
    tmp = LOCAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(acct))
    os.replace(tmp, LOCAL)


# ---------------------------------------------------------------- live loop

def tick(acct: dict | None) -> tuple[dict, bool]:
    """Fetch prices; on a new hourly close, forecast and trade. Returns the account and whether it traded."""
    from model4 import forecaster
    from model4.data import candles

    got = {s: candles(s, forecaster.CONTEXT) for s in SYMBOLS}
    prices = {s: p for s, (_, p) in got.items()}
    _live.update(prices=prices, at=datetime.now(timezone.utc))
    if acct is None:
        acct = {"started": datetime.now(timezone.utc).isoformat(), "last_bar": None,
                "books": {s: new_book() for s in SYMBOLS}}
        log.info(f"model 4 paper accounts started with ${START_CASH:.0f} per coin")
    # Bar times are open times; the bar just closed ends one hour later.
    last = max(df.index[-1] for df, _ in got.values()) + timedelta(hours=1)
    bar = last.isoformat()
    if acct["last_bar"] == bar:
        return acct, False
    frames = {s: df for s, (df, _) in got.items() if len(df) == forecaster.CONTEXT and df.index[-1] + timedelta(hours=1) == last}
    t0 = time.time()
    moves = forecaster.forecast(frames) if frames else {}
    log.info(f"model 4 forecast {len(moves)} coins in {time.time() - t0:.0f}s: "
             + ", ".join(f"{NAMES[s]} {m:+.2f}%" for s, m in moves.items()))
    traded = False
    for s in SYMBOLS:
        for msg in step(acct["books"][s], bar, prices[s], moves.get(s)):
            log.info(f"model 4 {NAMES[s]}: {msg}")
            traded = True
    acct["last_bar"] = bar
    return acct, traded


def run():
    DIR.mkdir(parents=True, exist_ok=True)
    acct = json.loads(LOCAL.read_text()) if (_pull() or LOCAL.exists()) else None
    pushed = 0.0
    while True:
        try:
            acct, traded = tick(acct)
            _save(acct)
            if traded or time.time() - pushed > 1800:
                _push()
                pushed = time.time()
        except Exception as e:  # network blips: log and retry next pass
            log.warning(f"model 4 tick failed: {e}")
        # Wake a little after each 5-minute mark, so a new hourly close is seen within ~6 minutes.
        time.sleep(300 - time.time() % 300 + 20)


# ---------------------------------------------------------------- page

def _account() -> dict | None:
    if not LOCAL.exists():
        return None
    try:
        return json.loads(LOCAL.read_text())
    except json.JSONDecodeError:  # mid-write on some filesystems; next refresh is fine
        return None


def summary(acct: dict | None = None) -> dict | None:
    acct = acct or _account()
    if acct is None:
        return None
    prices = _live.get("prices") or {}
    coins = {}
    for s, b in acct["books"].items():
        bal, nofee = mark(b, prices.get(s))
        n = b.get("n", len(b["trades"]))
        coins[s] = {"balance": round(bal, 4), "nofee": round(nofee, 4), "trades": n, "right": b["right"],
                    "open": b["pos"], "price": prices.get(s), "last_forecast": b["last_forecast"],
                    "recent": b["trades"][-10:]}
    total = sum(c["balance"] for c in coins.values())
    return {"started": acct["started"], "last_bar": acct["last_bar"], "start_cash": START_CASH * len(coins),
            "total": round(total, 4), "total_nofee": round(sum(c["nofee"] for c in coins.values()), 4),
            "trades": sum(c["trades"] for c in coins.values()), "right": sum(c["right"] for c in coins.values()),
            "coins": coins}


router = APIRouter()


@router.get("/api/model4")
def model4_api():
    return summary() or {"error": "model 4 not running yet"}


def _money(x) -> str:
    return "-" if x is None else f"${x:,.2f}"


def _cls(x, base=0.0) -> str:
    return "up" if x is not None and x >= base else "dn"


def _price(p) -> str:
    return "-" if p is None else (f"{p:,.1f}" if p >= 100 else f"{p:,.4f}")


def _hm(iso: str | None) -> str:
    return "-" if not iso else iso[5:16].replace("T", " ")


@router.get("/model4", response_class=HTMLResponse)
def model4_page():
    now = datetime.now(timezone.utc)
    s = summary()
    if s is None:
        body = "<section><p>Model 4 is starting up. Its first forecast lands within a few minutes.</p></section>"
    else:
        chg = s["total"] - s["start_cash"]
        rows, trades = "", []
        for sym, c in s["coins"].items():
            o = c["open"]
            pos = "-" if not o else f"{'Long' if o['side'] > 0 else 'Short'} from {_price(o['entry'])}, closes {_hm(o['close_at'])}"
            f = c["last_forecast"]
            fc = "-" if not f else f"<span class={_cls(f['move_pct'])}>{f['move_pct']:+.2f}%</span>"
            rows += (f"<tr><td>{NAMES.get(sym, sym)}</td><td>{_price(c['price'])}</td><td>{fc}</td>"
                     f"<td class={_cls(c['balance'], 100)}>{_money(c['balance'])}</td><td>{_money(c['nofee'])}</td>"
                     f"<td>{c['right']} of {c['trades']}</td><td>{pos}</td></tr>")
            trades += [{**t, "symbol": sym} for t in c["recent"]]
        trades.sort(key=lambda t: t["closed"], reverse=True)
        trows = "".join(
            f"<tr><td>{_hm(t['closed'])}</td><td>{NAMES.get(t['symbol'], t['symbol'])}</td><td>{'Long' if t['side'] > 0 else 'Short'}</td>"
            f"<td>{t['forecast']:+.2f}%</td><td class={_cls(t['ret_pct'])}>{t['ret_pct']:+.2f}%</td><td>{_money(t['balance'])}</td></tr>"
            for t in trades[:30])
        rate = f"right {s['right']} of {s['trades']}" if s["trades"] else "no closed trades yet"
        body = f"""<section>
<div class=big>{_money(s['total'])} <i class={_cls(chg)}>{chg:+.2f}</i></div>
<small>All 7 bots together, started with ${s['start_cash']:.0f} on {s['started'][:16].replace('T', ' ')} UTC. {rate}.
Without fees it would be {_money(s['total_nofee'])}. Last forecast after the {_hm(s['last_bar'])} UTC close.</small>
<h3>Each coin</h3>
<table><tr><th>Coin</th><th>Price</th><th>Next 4h forecast</th><th>Balance</th><th>No fees</th><th>Right</th><th>Open trade</th></tr>{rows}</table>
<p><small>It trades only when the forecast move is at least {THRESHOLD}%, because a round trip costs 0.2% in fees.
Each trade bets that coin's whole balance for {HOLD_HOURS} hours.</small></p>
<h3>Closed trades</h3>
<table><tr><th>Closed</th><th>Coin</th><th>Side</th><th>Forecast</th><th>Result</th><th>Balance</th></tr>{trows or '<tr><td colspan=6>None yet.</td></tr>'}</table>
</section>"""
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=300>
<title>Model 4 bot</title><style>
body{{font:15px system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#12161f;color:#eaecef}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}td,th{{padding:6px;border-bottom:1px solid #2b313a;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.up{{color:#0ecb81}}.dn{{color:#f6465d}}
.big{{font-size:28px;font-weight:600}}.big i{{font-size:16px;font-style:normal}}small{{color:#848e9c}}
section{{border:1px solid #2b313a;border-radius:10px;padding:4px 16px 12px;margin:16px 0;overflow-x:auto}}
a{{color:#fcd535}}h3{{margin:18px 0 4px;font-size:16px}}
</style></head><body>
<h1>Model 4: Kronos forecasts</h1>
<p><small>Fake $100 per coin, Binance spot prices, 0.1% fee each way. Every hour Kronos, a pretrained AI model for price charts,
guesses where each coin will be in 4 hours. Nothing was tuned on our data. See also <a href="/paper">model 1</a>,
<a href="/model2">model 2</a> and <a href="/model3">model 3</a>.</small></p>
{body}
<p><small>Now {now:%H:%M} UTC ({now + timedelta(hours=5):%H:%M} Pakistan). Page refreshes every 5 minutes.</small></p>
</body></html>"""


def start_thread():
    threading.Thread(target=run, daemon=True, name="model4").start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
