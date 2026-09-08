---
title: a5stocks backend
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

Order flow + continual-learning FastAPI backend for a5stocks.

`app.py` re-exports the FastAPI app from `main.py`. Models retrained at runtime
are written to `$MODEL_DIR`, which points at `/data` when the Space has
persistent storage enabled and `/tmp` otherwise.
