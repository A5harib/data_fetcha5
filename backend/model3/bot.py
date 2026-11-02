"""
Model 3 live: the weekly trend rotation (strategy.py) on Binance spot prices
with a fake $100, shown at /model3.

Every 5 minutes it reads the daily closes. Right after each 00:00 UTC close,
the sleeve whose weekday it is rebalances at the current price. In between it
only updates the account's value.

Memory: the account is saved locally and pushed to the PAPER_DATASET repo
under model3/ after every trade and every 30 minutes. A restart pulls it back.
Models 1 and 2's files in that dataset are never touched.

    MODEL3_BOT=1 python -m model3.bot      # run locally
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

from model3 import strategy as st
from model3.data import last_prices, recent
from model3.train import MODEL_DIR

DIR = Path(__file__).resolve().parent.parent / "data" / "model3"
LOCAL = DIR / "account.json"
REMOTE = "model3/account.json"
START_CASH = 100.0
NAMES = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB",
         "XRPUSDT": "XRP", "DOGEUSDT": "DOGE", "XAUUSDT": "Gold"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
log = logging.getLogger("model3.bot")
_live: dict = {}   # latest prices and closes, for the page


def _pull() -> bool:
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return False
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, REMOTE, repo_type="dataset")
        DIR.mkdir(parents=True, exist_ok=True)
        LOCAL.write_text(Path(p).read_text())
        log.info(f"restored model 3 from {repo}")
        return True
    except Exception as e:  # first run: nothing stored yet
        log.info(f"no model 3 state in {repo} yet ({type(e).__name__})")
        return False


def _push():
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    try:
        HfApi().create_commit(repo_id=repo, repo_type="dataset", commit_message="model 3 update",
                              operations=[CommitOperationAdd(REMOTE, str(LOCAL))])
    except Exception as e:  # never let storage stop trading; the next push retries
        log.warning(f"could not save model 3 to {repo}: {e}")


def _save(acct: dict):
    tmp = LOCAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(acct))
    os.replace(tmp, LOCAL)


def _books(acct: dict) -> list[st.Book]:
    return [st.Book.from_dict(b) for b in acct["books"]]


def _new_account(closes, prices) -> dict:
    books = st.new_account(START_CASH)
    # Start on the newest closed bar, so every sleeve buys its first picks now.
    for b in books:
        b.last_day = closes.index[-2].strftime("%Y-%m-%d")
    return {"started": datetime.now(timezone.utc).isoformat(), "start_prices": prices,
            "books": [json.loads(b.dumps()) for b in books]}


def tick(acct: dict | None, closes, prices) -> tuple[dict, bool]:
    """One pass of the live loop. Returns the account and whether it traded."""
    if acct is None:
        acct = _new_account(closes, prices)
        log.info(f"model 3 paper account started with ${START_CASH:.0f}")
    books = _books(acct)
    traded = False
    for b in books:
        traded |= st.step(b, closes, fill=prices, live=True)
    acct["books"] = [json.loads(b.dumps()) for b in books]
    return acct, traded


def run():
    DIR.mkdir(parents=True, exist_ok=True)
    acct = json.loads(LOCAL.read_text()) if (_pull() or LOCAL.exists()) else None
    pushed = 0.0
    while True:
        try:
            closes = recent(st.WARMUP + 30)
            prices = last_prices()
            _live.update(closes=closes, prices=prices, at=datetime.now(timezone.utc))
            before = None if acct is None else sum(len(b["trades"]) for b in acct["books"])
            acct, traded = tick(acct, closes, prices)
            if traded:
                n = sum(len(b["trades"]) for b in acct["books"]) - (before or 0)
                log.info(f"model 3 rebalanced ({n} trades), value ${summary(acct)['value']:.2f}")
            _save(acct)
            if traded or time.time() - pushed > 1800:
                _push()
                pushed = time.time()
        except Exception as e:  # network blips: log and retry next pass
            log.warning(f"model 3 tick failed: {e}")
        # Wake a little after each 5-minute mark, so a new daily close is seen within ~6 minutes.
        time.sleep(300 - time.time() % 300 + 20)


# ---------------------------------------------------------------- page

def _account() -> dict | None:
    if not LOCAL.exists():
        return None
    try:
        return json.loads(LOCAL.read_text())
    except json.JSONDecodeError:  # mid-write on some filesystems; next refresh is fine
        return None


def _meta() -> dict:
    p = MODEL_DIR / "meta.json"
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "models" / "model3" / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summary(acct: dict | None = None) -> dict | None:
    acct = acct or _account()
    if acct is None:
        return None
    books = _books(acct)
    prices = _live.get("prices") or {}
    if not prices:  # page served before the first live pass: value at the last close
        prices = {}
        for b in books:
            for t in b.trades:
                prices[t["symbol"]] = t["price"]
    qty, cash = {}, 0.0
    for b in books:
        cash += b.cash
        for s, q in b.qty.items():
            qty[s] = qty.get(s, 0.0) + q
    held = {s: q * prices.get(s, 0.0) for s, q in qty.items()}
    value = cash + sum(held.values())
    sp = acct["start_prices"]
    common = [s for s in sp if s in prices]
    trades = sorted((t for b in books for t in b.trades), key=lambda t: t["day"])
    return {
        "started": acct["started"], "start_cash": START_CASH, "value": round(value, 4), "cash": round(cash, 4),
        "holdings": {s: {"value": round(v, 4), "share": round(v / value, 4) if value else 0, "price": prices.get(s)}
                     for s, v in sorted(held.items(), key=lambda kv: -kv[1])},
        "fees_paid": round(sum(b.fees_paid for b in books), 4),
        "hold_btc": round(START_CASH * prices["BTCUSDT"] / sp["BTCUSDT"], 2) if "BTCUSDT" in common else None,
        "hold_all": round(START_CASH * sum(prices[s] / sp[s] for s in common) / len(common), 2) if common else None,
        "trades": len(trades), "recent_trades": trades[-40:],
        "last_day": max((b.last_day for b in books if b.last_day), default=None),
        "sleeves": [{"weekday": WEEKDAYS[b.weekday], "value": round(b.value(prices), 4) if prices else None,
                     "last_rebalance": b.last_rebalance} for b in books],
    }


router = APIRouter()


@router.get("/api/model3")
def model3_api():
    return summary() or {"error": "model 3 not running yet"}


def _money(x) -> str:
    return "-" if x is None else f"${x:,.2f}"


def _cls(x, base=0.0) -> str:
    return "up" if x is not None and x >= base else "dn"


def _price(p) -> str:
    return "-" if p is None else (f"{p:,.1f}" if p >= 100 else f"{p:,.4f}")


@router.get("/model3", response_class=HTMLResponse)
def model3_page():
    now = datetime.now(timezone.utc)
    s = summary()
    meta = _meta()
    if s is None:
        body = "<section><p>Model 3 is starting up. It buys its first picks within a few minutes.</p></section>"
    else:
        chg = s["value"] - s["start_cash"]
        hold_rows = "".join(
            f"<tr><td>{NAMES.get(k, k)}</td><td>{_price(v['price'])}</td><td>{_money(v['value'])}</td><td>{100 * v['share']:.0f}%</td></tr>"
            for k, v in s["holdings"].items())
        hold_rows += f"<tr><td>Cash (USDT)</td><td></td><td>{_money(s['cash'])}</td><td>{100 * s['cash'] / s['value']:.0f}%</td></tr>"
        closes = _live.get("closes")
        why = ""
        if closes is not None:
            picks = st.target_weights(closes)
            why_rows = "".join(
                f"<tr{' class=sel' if r['symbol'] in picks else ''}><td>{NAMES.get(r['symbol'], r['symbol'])}</td>"
                f"<td class={_cls(r['rise_pct'])}>{r['rise_pct']:+.1f}%</td><td class={_cls(r['vs_avg_pct'])}>{r['vs_avg_pct']:+.1f}%</td>"
                f"<td>{r['yearly_swing_pct']:.0f}%</td><td>{'buy ' + format(100 * picks[r['symbol']], '.0f') + '%' if r['symbol'] in picks else ('trending' if r['rise_pct'] > 0 and r['vs_avg_pct'] > 0 else 'no')}</td></tr>"
                for r in st.explain(closes))
            why = f"""<h3>What it sees at the last daily close</h3>
<table><tr><th>Asset</th><th>20-day move</th><th>vs 50-day average</th><th>Yearly swing</th><th>Pick</th></tr>{why_rows}</table>
<p><small>Trending means up over 20 days and above its 50-day average. It holds the two that rose most, smaller slices
for wilder coins, and keeps the rest in cash. "Buy" is the share of a sleeve that trades today.</small></p>"""
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        today_sleeve = WEEKDAYS[now.weekday() - 1 if now.weekday() else 6]
        sleeve_rows = "".join(f"<tr{' class=sel' if x['weekday'] == today_sleeve else ''}><td>{x['weekday']}</td><td>{_money(x['value'])}</td>"
                              f"<td>{x['last_rebalance'] or '-'}</td></tr>" for x in s["sleeves"])
        rows = "".join(f"<tr><td>{t['day']}</td><td>{t['side'].title()}</td><td>{NAMES.get(t['symbol'], t['symbol'])}</td>"
                       f"<td>{_price(t['price'])}</td><td>{_money(t['usd'])}</td></tr>" for t in reversed(s["recent_trades"]))
        body = f"""<section>
<div class=big>{_money(s['value'])} <i class={_cls(chg)}>{chg:+.2f}</i></div>
<small>Started $100 on {s['started'][:16].replace('T', ' ')} UTC. Fees paid so far: {_money(s['fees_paid'])}.</small>
<table><tr><th>Same $100 since the start</th><th>Now</th></tr>
<tr><td>Model 3</td><td class={_cls(s['value'], 100)}>{_money(s['value'])}</td></tr>
<tr><td>Just holding BTC</td><td class={_cls(s['hold_btc'], 100)}>{_money(s['hold_btc'])}</td></tr>
<tr><td>Holding all 7 equally</td><td class={_cls(s['hold_all'], 100)}>{_money(s['hold_all'])}</td></tr></table>
<h3>Holding now</h3>
<table><tr><th>Asset</th><th>Price</th><th>Value</th><th>Share</th></tr>{hold_rows}</table>
{why}
<h3>The 7 sleeves</h3>
<p><small>The $100 is split into 7 parts. Each one rebalances once a week on its own day, right after 00:00 UTC.
Next: {nxt:%a %H:%M} UTC.</small></p>
<table><tr><th>Trades on</th><th>Value</th><th>Last rebalance</th></tr>{sleeve_rows}</table>
<h3>Trades</h3>
<table><tr><th>Day</th><th>Side</th><th>Asset</th><th>Price</th><th>Amount</th></tr>{rows or '<tr><td colspan=5>None yet.</td></tr>'}</table>
</section>"""
    bt = ""
    if meta:
        u, a = meta["unseen"], meta["all"]
        yr = "".join(f"<tr><td>{y}</td><td class={_cls(v['ret_pct'])}>{v['ret_pct']:+.0f}%</td><td>{v['worst_drop_pct']:.0f}%</td><td>{v['months_up']}</td></tr>"
                     for y, v in a["years"].items())
        bt = f"""<section><h3>Backtest (same code, 0.1% fee each way)</h3>
<p>On {u['start']} to {u['end']}, data it was never tuned on: $100 became <b>{_money(u['per_100'])}</b>, worst drop {u['worst_drop_pct']:.0f}%.
Holding all 7 equally: {_money(u['hold_all_per_100'])}. Holding BTC: {_money(u['hold_btc_per_100'])}.</p>
<table><tr><th>Year</th><th>Return</th><th>Worst drop</th><th>Months up</th></tr>{yr}</table>
<p><small>It is a slow model. It still has down months and down years, and it shines when one or two coins run for weeks.
Judge it over months, not days.</small></p></section>"""
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=300>
<title>Model 3 bot</title><style>
body{{font:15px system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#12161f;color:#eaecef}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}td,th{{padding:6px;border-bottom:1px solid #2b313a;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.up{{color:#0ecb81}}.dn{{color:#f6465d}}
.big{{font-size:28px;font-weight:600}}.big i{{font-size:16px;font-style:normal}}small{{color:#848e9c}}
section{{border:1px solid #2b313a;border-radius:10px;padding:4px 16px 12px;margin:16px 0}}tr.sel td{{background:#1b2130}}
a{{color:#fcd535}}h3{{margin:18px 0 4px;font-size:16px}}
</style></head><body>
<h1>Model 3: weekly trend rotation</h1>
<p><small>Fake $100, Binance spot prices, 0.1% fee each way. It holds the 2 strongest of BTC, ETH, SOL, BNB, XRP, DOGE and gold,
or cash when nothing is rising. See also <a href="/paper">model 1</a> and <a href="/model2">model 2</a>.</small></p>
{body}{bt}
<p><small>Now {now:%H:%M} UTC ({now + timedelta(hours=5):%H:%M} Pakistan). Page refreshes every 5 minutes.</small></p>
</body></html>"""


def start_thread():
    threading.Thread(target=run, daemon=True, name="model3").start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
