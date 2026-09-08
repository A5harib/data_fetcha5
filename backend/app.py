"""
HuggingFace Spaces entrypoint.

Free-tier Gradio Spaces run on ZeroGPU, which aborts startup unless the
`spaces` package completes a handshake. That handshake is triggered from a
monkeypatched `gradio.Blocks.launch` (see spaces/zero/__init__.py), so the
process must actually launch a Gradio app, and it must own at least one
@spaces.GPU function. Serving uvicorn directly gets the container killed a
few seconds after startup.

So: launch a minimal Gradio status page and mount the real FastAPI app under
it. Gradio serves /, FastAPI serves /health, /api/* and /ws/*.
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

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from main import app as fastapi_app  # noqa: E402

# ZeroGPU refuses to start a Space with no GPU-decorated function. This
# backend is CPU-only (xgboost-cpu, sklearn) and never calls this.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_probe():  # pragma: no cover
        return "ok"
except ImportError:
    pass

# Seed the repo's baked-in models into MODEL_DIR on first boot so a cold start
# predicts from the trained artifacts instead of nothing.
_seed = Path(__file__).resolve().parent / "models"
_target = Path(os.environ["MODEL_DIR"])
_target.mkdir(parents=True, exist_ok=True)
for f in _seed.glob("*"):
    dest = _target / f.name
    if f.is_file() and not dest.exists():
        dest.write_bytes(f.read_bytes())

with gr.Blocks(title="a5stocks backend") as demo:
    gr.Markdown(
        "# a5stocks backend\n"
        "Order flow + continual-learning API. This page is only here because "
        "the Space runtime requires a Gradio app; the service itself is the "
        "mounted FastAPI application.\n\n"
        "- `GET /health`\n"
        "- `GET /api/analytics/{symbol}`\n"
        "- `WS  /ws/analytics/{symbol}`\n"
        "- `GET /docs`"
    )

# The Gradio app exists only to satisfy ZeroGPU, which fires its startup
# handshake from a monkeypatched gradio.Blocks.launch (spaces/zero/__init__.py).
# Launch it on a side port with prevent_thread_lock so the hook runs and returns
# immediately. It is deliberately NOT mounted into FastAPI: gr.mount_gradio_app
# takes over the root path and shadows /health, /api/* and /ws/*.
try:
    demo.launch(
        prevent_thread_lock=True,
        share=False,
        quiet=True,
        server_port=7999,
    )
except Exception as exc:  # pragma: no cover
    print(f"gradio launch hook failed (continuing): {exc}")

# The Space proxies $PORT, so the API owns it.
uvicorn.run(fastapi_app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
