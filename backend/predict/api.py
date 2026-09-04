"""
FastAPI routes for the short-horizon predictor.

Mounted as a router so the existing reversal system keeps running untouched
while this one is evaluated side by side.

    # in main.py
    from predict.api import router as predict_router, attach_recorder
    app.include_router(predict_router)

Every response carries `ready`. When there is no trained model, or the market
state is too thin to describe, that flag is False and the UI must show
"no prediction" rather than a number. A dashboard that renders an arrow when
the model has no opinion is worse than one that shows nothing.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from predict.features import FEATURE_NAMES
from predict.model import Predictor
from predict.recorder import MarketRecorder

logger = logging.getLogger("predict.api")

router = APIRouter(prefix="/api/predict", tags=["short-horizon"])

# One predictor and one recorder per symbol. The recorder doubles as the
# live feature source, so training data and live inference come from exactly
# the same code path. That equivalence is what the old design lacked: it
# trained on a volume proxy and predicted on real book data.
_predictors: Dict[str, Predictor] = {}
_recorders: Dict[str, MarketRecorder] = {}


def get_predictor(symbol: str) -> Predictor:
    sym = symbol.upper()
    if sym not in _predictors:
        p = Predictor(symbol=sym)
        p.load()  # A missing model leaves it unready, which is a valid state.
        _predictors[sym] = p
    return _predictors[sym]


def attach_recorder(symbol: str, recorder: MarketRecorder):
    """Register the live recorder feeding a symbol, so predictions can read its state."""
    _recorders[symbol.upper()] = recorder


def get_recorder(symbol: str) -> Optional[MarketRecorder]:
    return _recorders.get(symbol.upper())


def _current_features(symbol: str) -> Optional[dict]:
    """Build a feature snapshot from live state, or None if the state is too thin."""
    rec = get_recorder(symbol)
    if rec is None:
        return None
    snap = rec.tick()
    if snap is None:
        # tick() returns None within the same second, so fall back to
        # building one directly rather than reporting a false outage.
        snap = rec.tick(now_ms=int(time.time() * 1000) + 1000)
    return snap.features if snap else None


@router.get("/{symbol}")
async def get_prediction(symbol: str):
    """Current prediction for the next N seconds."""
    predictor = get_predictor(symbol)
    feats = _current_features(symbol)

    if feats is None:
        return {
            "symbol": symbol.upper(),
            "ready": False,
            "direction": "WARMING_UP",
            "reason": "Not enough live market history yet. Needs 120s of continuous data.",
            "score": 0.0,
            "expected_move_pct": 0.0,
            "confidence": 0.0,
        }

    out = predictor.predict(feats)
    out["symbol"] = symbol.upper()
    out["timestamp_ms"] = int(time.time() * 1000)
    return out


@router.get("/{symbol}/status")
async def get_status(symbol: str):
    """Model metadata: is it trained, how well did it score, when was it fit."""
    predictor = get_predictor(symbol)
    rec = get_recorder(symbol)
    status = predictor.status()
    status["recording"] = rec is not None
    status["snapshots_recorded"] = rec.snapshots_written if rec else 0
    status["feature_names"] = FEATURE_NAMES
    return status


@router.get("/{symbol}/features")
async def get_features(symbol: str):
    """Raw current feature values. Useful for debugging a prediction you distrust."""
    feats = _current_features(symbol)
    if feats is None:
        raise HTTPException(status_code=503, detail="Market state not ready. Needs 120s of history.")
    return {"symbol": symbol.upper(), "timestamp_ms": int(time.time() * 1000), "features": feats}


@router.websocket("/ws/{symbol}")
async def prediction_stream(websocket: WebSocket, symbol: str):
    """Push a prediction once per second."""
    await websocket.accept()
    sym = symbol.upper()
    predictor = get_predictor(sym)
    logger.info(f"Prediction stream opened for {sym}")

    try:
        while True:
            feats = _current_features(sym)
            if feats is None:
                payload = {
                    "symbol": sym,
                    "ready": False,
                    "direction": "WARMING_UP",
                    "score": 0.0,
                    "expected_move_pct": 0.0,
                    "confidence": 0.0,
                }
            else:
                payload = predictor.predict(feats)
                payload["symbol"] = sym
            payload["timestamp_ms"] = int(time.time() * 1000)

            await websocket.send_json(payload)
            import asyncio

            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info(f"Prediction stream closed for {sym}")
    except Exception as e:
        logger.error(f"Prediction stream error for {sym}: {e}")
