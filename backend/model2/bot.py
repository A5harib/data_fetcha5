"""
Model 2 live: run the self-learning agent on Binance spot prices with a fake
$100, and show it next to model 1 at /model2.

Every closed minute the agent scores the market, its four virtual traders move,
and the paper account follows whichever one has made the most money lately.
Once an hour it learns from the minutes whose outcome is now known. So the
model on the page keeps changing as it runs.

Memory: the agent (weights, pickiness levels, paper account) is saved locally
every minute and pushed to the PAPER_DATASET repo under model2/ after every
trade and every 30 minutes. A restart pulls it back, then replays the minutes
it missed (up to about two days) before trading again. Model 1's files in that
dataset are never touched.

    MODEL2_BOT=1 python -m model2.bot --symbol BTCUSDT    # run one bot locally
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from forecast.data import LiveFeed
from model2.agent import FEE, LEVELS, LOOKBACK_MIN, WARMUP, Agent, advance, features
from model2.train import MODEL_DIR

DIR = Path(__file__).resolve().parent.parent / "data" / "model2"
SYMBOLS = [s.strip().upper() for s in os.environ.get("MODEL2_SYMBOLS", "BTCUSDT,XAUUSDT").split(",") if s.strip()]
NAMES = {"BTCUSDT": "BTC", "XAUUSDT": "Gold"}
START_CASH = 100.0
# Enough 1m bars for the 30-day trend features plus two days of catch-up after a restart.
FEED_BARS = WARMUP + 2 * 1440 + 120
log = logging.getLogger("model2.bot")


def _local(sym: str) -> Path:
    return DIR / f"{sym}_agent.json"


def _remote(sym: str) -> str:
    return f"model2/{sym}_agent.json"


def _pull(sym: str) -> bool:
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return False
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, _remote(sym), repo_type="dataset")
        DIR.mkdir(parents=True, exist_ok=True)
        _local(sym).write_text(Path(p).read_text())
        log.info(f"{sym}: restored model 2 from {repo}")
        return True
    except Exception as e:  # first run: nothing stored yet
        log.info(f"{sym}: no model 2 state in {repo} yet ({type(e).__name__})")
        return False


def _push(sym: str):
    repo = os.environ.get("PAPER_DATASET")
    if not repo:
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    try:
        HfApi().create_commit(repo_id=repo, repo_type="dataset", commit_message=f"model 2 {sym} update",
                              operations=[CommitOperationAdd(_remote(sym), str(_local(sym)))])
    except Exception as e:  # never let storage stop trading; the next push retries
        log.warning(f"{sym}: could not save model 2 to {repo}: {e}")


def _save(sym: str, agent: Agent):
    p = _local(sym)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(agent.dumps())
    os.replace(tmp, p)


def _load_agent(sym: str) -> Agent:
    if _pull(sym) or _local(sym).exists():
        return Agent.from_dict(json.loads(_local(sym).read_text()))
    seed = MODEL_DIR / f"{sym}.json"
    if not seed.exists():
        raise FileNotFoundError(f"no trained model 2 for {sym}: python -m model2.train --symbol {sym}")
    log.info(f"{sym}: starting from the trained model ({seed})")
    return Agent.from_dict(json.loads(seed.read_text()))


def run(sym: str):
    DIR.mkdir(parents=True, exist_ok=True)
    agent = _load_agent(sym)
    feed = LiveFeed(sym, minutes=FEED_BARS, source=os.environ.get("MODEL2_SOURCE", "spot"))
    pushed = 0.0

    def on_event(ev, bar):
        late = datetime.now(timezone.utc) - bar.to_pydatetime() > timedelta(minutes=5)
        if "opened" in ev:
            p = ev["opened"]
            log.info(f"{sym} model 2 opened {'LONG' if p['side'] > 0 else 'SHORT'} at {p['entry']} "
                     f"(level top {100 * (1 - p['level']):.1f}%){' [catch-up]' if late else ''}")
        if "closed" in ev:
            t = ev["closed"]
            if late:
                t["catch_up"] = True
            log.info(f"{sym} model 2 closed {t['move_pct']:+.3f}%, balance ${t['cash']:.2f}")

    while True:
        try:
            df = feed.get()
            before = agent.last_bar
            n_trades = len(agent.real["trades"]) if agent.real else 0
            had_pos = bool(agent.real and agent.real["position"])
            advance(agent, df, features(df, sym), on_event)
            if agent.real is None:
                # First live start: catch-up is done, so the paper account starts now.
                agent.start_real(START_CASH, datetime.now(timezone.utc).isoformat())
                log.info(f"{sym}: model 2 paper account started with ${START_CASH:.0f}")
            if agent.last_bar != before:
                _save(sym, agent)
                traded = len(agent.real["trades"]) != n_trades or bool(agent.real["position"]) != had_pos
                if traded or time.time() - pushed > 1800:
                    _push(sym)
                    pushed = time.time()
        except Exception as e:  # network blips: log and retry next minute
            log.warning(f"{sym} model 2 tick failed: {e}")
        time.sleep(66 - time.time() % 60)


# ---------------------------------------------------------------- page

def _state(sym: str) -> dict | None:
    p = _local(sym)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:  # mid-write on some filesystems; next refresh is fine
        return None


def _meta(sym: str) -> dict:
    p = MODEL_DIR / f"{sym}_meta.json"
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "models" / "model2" / f"{sym}_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _level_name(q) -> str:
    return f"top {100 * (1 - float(q)):g}%"


def summary(sym: str) -> dict | None:
    st = _state(sym)
    if st is None or st.get("real") is None:
        return None
    a = Agent.from_dict(st)
    r = a.real
    now = a.last_bar or 0
    tr = r["trades"]
    lv = {}
    for q, L in a.levels.items():
        recent = [n for t, n in L["trades"] if t >= now - LOOKBACK_MIN]
        grow = 1.0
        for n in recent:
            grow *= 1 + n
        lv[_level_name(q)] = {"trades_30d": len(recent), "right_30d": (sum(n > -2 * FEE for n in recent) / len(recent)) if recent else None,
                              "per_100_30d": round(100 * grow, 2), "in_trade": bool(L["pos"]),
                              "live": L.get("live", {"n": 0, "right": 0, "growth": 1.0})}
    best = a.best_level(now + 1)
    return {
        "symbol": sym, "started": r["started"], "start_cash": r["start_cash"],
        "cash": round(r["cash"], 4), "cash_no_fees": round(r["cash_no_fees"], 4),
        "trades": len(tr), "right": sum(t["right"] for t in tr), "position": r["position"],
        "following": _level_name(best) if best is not None else None, "levels": lv,
        "hours_learned": a.hours_learned, "last_bar": (datetime.fromtimestamp(60 * a.last_bar, timezone.utc).isoformat()
                                                        if a.last_bar else None),
        "score": None if a.score is None else round(a.score, 4),
        "recent_trades": tr[-50:],
    }


def model1(sym: str, since: str) -> dict | None:
    """Model 1's paper results, overall and over the same window as model 2."""
    try:
        from forecast import paper
        st = paper.load(sym)
    except Exception:
        return None
    if st is None:
        return None
    tr = st["trades"]
    same = [t for t in tr if t["entry_time"] >= since[:19]]
    grow = 1.0
    for t in same:
        grow *= (1 - FEE) ** 2 * (1 + t["move_pct"] / 100)
    return {"cash": st["cash"], "start_cash": st["start_cash"], "trades": len(tr), "right": sum(t["right"] for t in tr),
            "same_window": {"trades": len(same), "right": sum(t["right"] for t in same), "per_100": round(100 * grow, 2)}}


router = APIRouter()


@router.get("/api/model2")
def model2_all():
    return {sym: summary(sym) for sym in SYMBOLS}


@router.get("/api/model2/{symbol}")
def model2_one(symbol: str):
    return summary(symbol.upper()) or {"error": "model 2 not running yet"}


def _pct(a, b) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def _pct1(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _t(minute) -> str:
    return datetime.fromtimestamp(60 * minute, timezone.utc).strftime("%m-%d %H:%M")


def _price(p: float) -> str:
    return f"{p:,.1f}" if p >= 100 else f"{p:,.4f}"


def _card(sym: str) -> str:
    s = summary(sym)
    name = NAMES.get(sym, sym)
    if s is None:
        return f"<section><h2>{name}</h2><p>Model 2 is starting up. It replays the last day or so before its first trade.</p></section>"
    chg = s["cash"] - s["start_cash"]
    m1 = model1(sym, s["started"])
    pos = s["position"]
    if pos:
        now_line = (f"Open: <b>{'Long' if pos['side'] > 0 else 'Short'}</b> at {_price(pos['entry'])} since {_t(pos['entry_t'])} UTC. "
                    f"It holds while the signal agrees, 30 minutes to 6 hours.")
    elif s["following"]:
        now_line = f"No open trade. Following the {s['following']} style, waiting for a strong enough signal."
    else:
        now_line = "Sitting out: none of its four styles made money after fees in the last 30 days."
    if m1:
        w = m1["same_window"]
        cmp_rows = (f"<tr><td>Model 2 (self-learning)</td><td>{s['trades']}</td><td>{_pct(s['right'], s['trades'])}</td>"
                    f"<td class={'up' if chg >= 0 else 'dn'}>${s['cash']:.2f}</td></tr>"
                    f"<tr><td>Model 1 (snap-back), same period</td><td>{w['trades']}</td><td>{_pct(w['right'], w['trades'])}</td>"
                    f"<td class={'up' if w['per_100'] >= 100 else 'dn'}>${w['per_100']:.2f}</td></tr>"
                    f"<tr><td><small>Model 1, whole run so far</small></td><td>{m1['trades']}</td><td>{_pct(m1['right'], m1['trades'])}</td>"
                    f"<td class={'up' if m1['cash'] >= m1['start_cash'] else 'dn'}>${m1['cash']:.2f}</td></tr>")
    else:
        cmp_rows = (f"<tr><td>Model 2 (self-learning)</td><td>{s['trades']}</td><td>{_pct(s['right'], s['trades'])}</td>"
                    f"<td>${s['cash']:.2f}</td></tr><tr><td colspan=4><small>Model 1 results not available here.</small></td></tr>")
    lv_rows = "".join(
        f"<tr{' class=sel' if k == s['following'] else ''}><td>{k}</td><td>{v['trades_30d']}</td>"
        f"<td>{_pct1(v['right_30d'])}</td>"
        f"<td class={'up' if v['per_100_30d'] >= 100 else 'dn'}>${v['per_100_30d']:.2f}</td>"
        f"<td>{v['live']['n']}</td><td>{_pct(v['live']['right'], v['live']['n'])}</td>"
        f"<td class={'up' if v['live']['growth'] >= 1 else 'dn'}>${100 * v['live']['growth']:.2f}</td></tr>"
        for k, v in s["levels"].items())
    rows = "".join(
        f"<tr><td>{_t(t['entry_t'])}</td><td>{'Long' if t['side'] > 0 else 'Short'}</td><td>{(t['exit_t'] - t['entry_t']) // 60}h{(t['exit_t'] - t['entry_t']) % 60:02d}</td>"
        f"<td>{_price(t['entry'])}</td><td>{_price(t['exit'])}</td>"
        f"<td class={'up' if t['right'] else 'dn'}>{t['move_pct']:+.2f}%</td><td>${t['cash']:.2f}</td></tr>"
        for t in reversed(s["recent_trades"]))
    bt = _meta(sym).get("backtest", {})
    bt_line = ", ".join(f"{y}: right {v['right']:.0%}, $100 to ${v['per_100']:.0f}" for y, v in bt.items())
    return f"""<section><h2>{name}</h2>
<div class=big>${s['cash']:.2f} <i class={'up' if chg >= 0 else 'dn'}>{chg:+.2f}</i></div>
<small>Started ${s['start_cash']:.0f} on {s['started'][:16].replace('T', ' ')} UTC. Without fees: ${s['cash_no_fees']:.2f}.</small>
<p>{now_line}</p>
<table><tr><th>Model</th><th>Trades</th><th>Right</th><th>$100 is now</th></tr>{cmp_rows}</table>
<h3>What it has learned</h3>
<p><small>It has learned from {s['hours_learned']:,} hours of market so far and adds one more every hour.
The paper account copies whichever style below made the most after fees over the last 30 days (highlighted),
and sits out when none of them made money.</small></p>
<table><tr><th rowspan=2>Style (how picky)</th><th colspan=3>Last 30 days</th><th colspan=3>Since going live</th></tr>
<tr><th>Calls</th><th>Right</th><th>$100 would be</th><th>Calls</th><th>Right</th><th>$100 would be</th></tr>{lv_rows}</table>
<p><small>Each style trades on paper all the time, so these columns show model 2's calls even while the account sits out.
"Last 30 days" includes the replay of the days before it went live.</small></p>
<h3>Trades</h3>
<table><tr><th>Opened (UTC)</th><th>Side</th><th>Held</th><th>Entry</th><th>Exit</th><th>Move</th><th>Balance</th></tr>{rows or '<tr><td colspan=7>None yet.</td></tr>'}</table>
<p><small>Backtest on past years (same code, learning as it went): {bt_line or 'n/a'}.</small></p>
</section>"""


@router.get("/model2", response_class=HTMLResponse)
def model2_page():
    now = datetime.now(timezone.utc)
    cards = "".join(_card(s) for s in SYMBOLS)
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><meta http-equiv=refresh content=60>
<title>Model 2 bots</title><style>
body{{font:15px system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#12161f;color:#eaecef}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}td,th{{padding:6px;border-bottom:1px solid #2b313a;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.up{{color:#0ecb81}}.dn{{color:#f6465d}}
.big{{font-size:28px;font-weight:600}}.big i{{font-size:16px;font-style:normal}}small{{color:#848e9c}}
section{{border:1px solid #2b313a;border-radius:10px;padding:4px 16px 12px;margin:16px 0}}tr.sel td{{background:#1b2130}}
a{{color:#fcd535}}h3{{margin:18px 0 4px;font-size:16px}}
</style></head><body>
<h1>Model 2: self-learning bots</h1>
<p><small>Fake $100 each, Binance spot prices, 0.06% fee each way, same rules as <a href="/paper">model 1</a>.
They learn from the market every hour while they run.</small></p>
{cards}
<p><small>Now {now:%H:%M} UTC ({now + timedelta(hours=5):%H:%M} Pakistan). Page refreshes every minute.</small></p>
</body></html>"""


# ---------------------------------------------------------------- startup

def start_threads():
    for sym in SYMBOLS:
        threading.Thread(target=run, args=(sym,), daemon=True, name=f"model2-{sym}").start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(a.symbol.upper())
