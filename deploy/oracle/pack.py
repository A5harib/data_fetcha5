"""
Build paperbot.zip: the bot, the trained BTC model, and the current run's
balance and trade history, ready to copy to the server.

    python deploy/oracle/pack.py
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "paperbot.zip"

files = [(p, p.relative_to(REPO)) for p in (REPO / "backend/forecast").glob("*.py")]
files += [(p, p.relative_to(REPO)) for p in (REPO / "backend/models/forecast/BTCUSDT_60m").iterdir()]
# Carry over the running balance so the server continues the same run.
for name in ("BTCUSDT_state.json", "BTCUSDT_trades.jsonl"):
    p = REPO / "backend/data/paper" / name
    if p.exists():
        files.append((p, p.relative_to(REPO)))
files += [(HERE / n, Path(n)) for n in ("setup.sh", "paperbot.service", "requirements-paper.txt")]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in files:
        if src.suffix in (".sh", ".service", ".txt"):
            # Windows line endings break bash and systemd.
            z.writestr(arc.as_posix(), src.read_bytes().replace(b"\r\n", b"\n"))
        else:
            z.write(src, arc.as_posix())
print(f"Built {OUT} ({len(files)} files)")
