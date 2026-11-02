"""
Kronos-small forecasts for model 4.

Kronos is a pretrained model for price candles (NeoQuasar/Kronos-small on the
Hugging Face Hub). It is used as-is, with no training on our data. Given the
last CONTEXT hourly candles it draws SAMPLES possible next HORIZON hours and
averages them. The forecast move is that average's last close against the
last real close.

Runs on CPU. On the Space, torch must never touch the ZeroGPU device from a
background thread, so the device is pinned to "cpu".
"""
from __future__ import annotations

import logging
import threading

import pandas as pd

log = logging.getLogger("model4.forecaster")
CONTEXT = 400      # hourly bars fed in (the model's limit is 512)
HORIZON = 4        # hours ahead, which is also how long a trade is held
SAMPLES = 8        # forecasts averaged per coin
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"

_lock = threading.Lock()
_predictor = None


def _load():
    global _predictor
    with _lock:
        if _predictor is None:
            import torch

            from model4.kronos import Kronos, KronosPredictor, KronosTokenizer

            torch.set_num_threads(2)  # leave CPU for the other bots
            tok = KronosTokenizer.from_pretrained(TOKENIZER)
            model = Kronos.from_pretrained(MODEL)
            tok.eval()
            model.eval()
            _predictor = KronosPredictor(model, tok, device="cpu", max_context=512)
            log.info(f"loaded {MODEL}")
    return _predictor


def forecast(frames: dict[str, pd.DataFrame], seed: int | None = None) -> dict[str, float]:
    """Forecast move in % over the next HORIZON hours for each symbol. All frames must have the same length."""
    import torch

    p = _load()
    if seed is not None:
        torch.manual_seed(seed)
    syms = list(frames)
    xs, ys = [], []
    for s in syms:
        idx = frames[s].index
        xs.append(pd.Series(idx))
        ys.append(pd.Series(pd.date_range(idx[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="h")))
    with _lock, torch.no_grad():
        preds = p.predict_batch([frames[s].reset_index(drop=True) for s in syms], xs, ys, pred_len=HORIZON,
                                T=1.0, top_p=0.9, sample_count=SAMPLES, verbose=False)
    return {s: float(100 * (pr["close"].iloc[-1] / frames[s]["close"].iloc[-1] - 1)) for s, pr in zip(syms, preds)}
