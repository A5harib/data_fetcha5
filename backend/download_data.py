"""Download 1m futures klines from Binance, paginating backwards past the 1500-bar cap."""
import json, urllib.request, time, sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)


def fetch(sym, end_ms=None, limit=1500):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&limit={limit}"
    if end_ms:
        url += f"&endTime={end_ms}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def download(sym, target_bars):
    """Paginate backwards, checkpointing to disk so a dropped connection near the
    end does not discard everything fetched so far."""
    path = DATA_DIR / f"{sym}_1m.json"
    out, end = [], None
    try:
        while len(out) < target_bars:
            batch = fetch(sym, end)
            if not batch:
                break
            out = batch + out
            end = batch[0][0] - 1
            oldest = time.strftime('%Y-%m-%d %H:%M', time.gmtime(batch[0][0] / 1000))
            print(f"  {sym}: {len(out)} bars, oldest={oldest}", flush=True)
            path.write_text(json.dumps(out))  # checkpoint
            time.sleep(0.5)
            if len(batch) < 1500:
                break
    except Exception as e:
        print(f"  {sym}: stopped early ({type(e).__name__}: {e}); keeping {len(out)} bars", flush=True)
    if out:
        path.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    for sym in ["BTCUSDT", "XAUUSDT"]:
        d = download(sym, target)
        p = DATA_DIR / f"{sym}_1m.json"
        print(f"SAVED {sym}: {len(d)} bars -> {p} ({p.stat().st_size/1e6:.1f} MB)", flush=True)
