"""
HuggingFace Spaces entrypoint (Gradio SDK).

The Docker SDK is a paid feature, so the Space runs this file instead of a
Dockerfile. Gradio's runner imports `app.py` and serves whatever ASGI app it
finds, so re-exporting the FastAPI app from main.py is all that is needed.
"""
import os
from pathlib import Path

# Spaces gives a writable persistent mount at /data when storage is enabled.
# Without it, fall back to a writable tmp dir: retraining still works for the
# life of the container, models just do not survive a restart.
if "MODEL_DIR" not in os.environ:
    persistent = Path("/data")
    base = persistent if persistent.is_dir() and os.access(persistent, os.W_OK) else Path("/tmp")
    os.environ["MODEL_DIR"] = str(base / "models")

from main import app  # noqa: E402

# Seed the repo's baked-in models into MODEL_DIR on first boot so a cold start
# predicts from the trained artifacts instead of nothing.
_seed = Path(__file__).resolve().parent / "models"
_target = Path(os.environ["MODEL_DIR"])
_target.mkdir(parents=True, exist_ok=True)
for f in _seed.glob("*"):
    dest = _target / f.name
    if f.is_file() and not dest.exists():
        dest.write_bytes(f.read_bytes())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
