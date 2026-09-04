"""
FastAPI Real-Time Order Flow & Continual ML Streaming Server.

Provides:
- WebSocket endpoint: /ws/analytics/{symbol}
- Continual Online Learning Engine (experience replay & atomic background retraining)
- REST status & continual learning monitoring endpoints: /health, /api/analytics/{symbol}, /api/continual-stats/{symbol}
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from engine.analytics import OrderFlowAnalytics
from engine.binance_stream import BinanceFuturesStreamConsumer
from engine.orderflow_buffer import Candle, OrderFlowBuffer
from ml.continual_learner import ContinualLearningEngine
from ml.predictor import ReversalPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")


class SymbolSession:
    """Manages ingestion, analytics, continuous learning, and multi-client WebSockets per symbol."""
    def __init__(self, symbol: str, predictor: ReversalPredictor):
        self.symbol = symbol.upper()
        self.buffer = OrderFlowBuffer(symbol=self.symbol)
        self.consumer = BinanceFuturesStreamConsumer(symbol=self.symbol, buffer=self.buffer)
        self.analytics = OrderFlowAnalytics(buffer=self.buffer)
        self.predictor = predictor
        
        # Continual Online Learning Engine
        self.continual_learner = ContinualLearningEngine(
            predictor=self.predictor,
            retrain_batch_threshold=15,  # Retrain every 15 newly resolved candles
            max_replay_size=3000
        )
        
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task: asyncio.Task | None = None
        self._last_closed_candle_time: int = 0

        # Hook into buffer updates
        self.consumer.register_on_tick_callback(self._on_tick)

    def _on_tick(self):
        """Called on every market tick to check for candle completions for continual learning."""
        if self.buffer.candles_1m:
            last_closed = self.buffer.candles_1m[-1]
            if last_closed.open_time_ms != self._last_closed_candle_time:
                self._last_closed_candle_time = last_closed.open_time_ms
                indicators = self.analytics.calculate_indicators()
                recent_candles = list(self.buffer.candles_1m)
                
                # Notify continual learning engine
                self.continual_learner.on_candle_closed(
                    closed_candle=last_closed,
                    features_dict=indicators,
                    recent_candles=recent_candles
                )

    async def start(self):
        """Start consumer and broadcast loop."""
        await self.consumer.start()
        if self.broadcast_task is None or self.broadcast_task.done():
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self):
        """Stop consumer and broadcast task."""
        await self.consumer.stop()
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass

    def build_payload(self) -> dict:
        """Construct the exact JSON payload expected by the frontend."""
        indicators = self.analytics.calculate_indicators()
        div_result = self.analytics.detect_divergence()
        
        # ML Inference
        prediction = self.predictor.predict(
            indicator_dict=indicators,
            active_factors=div_result.factors
        )

        current_price = self.buffer.last_price
        if current_price == 0.0 and self.buffer.current_candle:
            current_price = self.buffer.current_candle.close

        payload = {
            "symbol": self.symbol,
            "price": round(current_price, 2),
            "cvd_1m": round(self.buffer.cvd_1m, 2),
            "orderbook_imbalance_pct": round(self.buffer.orderbook_imbalance_pct, 1),
            "reversal_prediction": {
                "probability": prediction["probability"],
                "signal": prediction["signal"],
                "factors": prediction["factors"]
            }
        }
        return payload

    async def _broadcast_loop(self):
        """Broadcast live signals at regular intervals to all connected UI clients."""
        logger.info(f"Broadcast loop active for {self.symbol}")
        while True:
            try:
                await asyncio.sleep(settings.BROADCAST_INTERVAL_SEC)
                if not self.active_connections:
                    continue

                payload = self.build_payload()
                json_str = json.dumps(payload)

                disconnected = set()
                for ws in self.active_connections:
                    try:
                        await ws.send_text(json_str)
                    except Exception:
                        disconnected.add(ws)

                for ws in disconnected:
                    self.active_connections.discard(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop for {self.symbol}: {e}")
                await asyncio.sleep(1.0)


class StreamManager:
    def __init__(self):
        # One predictor per symbol: each loads its own model + scaler.
        self.predictors: Dict[str, ReversalPredictor] = {}
        self.sessions: Dict[str, SymbolSession] = {}

    def _predictor_for(self, sym: str) -> ReversalPredictor:
        if sym not in self.predictors:
            self.predictors[sym] = ReversalPredictor(symbol=sym)
        return self.predictors[sym]

    async def get_or_create_session(self, symbol: str) -> SymbolSession:
        sym = symbol.upper()
        if sym not in self.sessions:
            session = SymbolSession(symbol=sym, predictor=self._predictor_for(sym))
            await session.start()
            self.sessions[sym] = session
        return self.sessions[sym]

    async def cleanup_session_if_idle(self, symbol: str):
        sym = symbol.upper()
        if sym in self.sessions:
            session = self.sessions[sym]
            if len(session.active_connections) == 0:
                logger.info(f"No clients connected to {sym}. Stopping stream...")
                await session.stop()
                del self.sessions[sym]

    async def shutdown(self):
        for session in self.sessions.values():
            await session.stop()
        self.sessions.clear()


stream_manager = StreamManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Quantitative Order Flow Server...")
    yield
    logger.info("Shutting down server and streams...")
    await stream_manager.shutdown()


app = FastAPI(
    title="Crypto Order Flow & ML Continual Learning Backend",
    description="Real-time Binance Order Flow Engine with CVD Divergence and Continual Learning XGBoost Model",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "active_symbols": list(stream_manager.sessions.keys()),
        "model_loaded": stream_manager.predictor.model is not None
    }


@app.get("/api/analytics/{symbol}")
async def get_symbol_analytics(symbol: str):
    session = await stream_manager.get_or_create_session(symbol.upper())
    return session.build_payload()


@app.get("/api/continual-stats/{symbol}")
async def get_continual_learning_stats(symbol: str):
    session = await stream_manager.get_or_create_session(symbol.upper())
    return {
        "symbol": session.symbol,
        "continual_learning": session.continual_learner.get_status()
    }


@app.post("/api/ml/trigger-retrain/{symbol}")
async def manual_trigger_retrain(symbol: str):
    session = await stream_manager.get_or_create_session(symbol.upper())
    asyncio.create_task(session.continual_learner.trigger_incremental_retrain())
    return {"message": f"Triggered manual continual retrain for {session.symbol}"}


@app.websocket("/ws/analytics/{symbol}")
async def websocket_analytics_endpoint(websocket: WebSocket, symbol: str):
    symbol = symbol.upper()
    await websocket.accept()
    logger.info(f"WebSocket client connected for {symbol}")

    session = await stream_manager.get_or_create_session(symbol)
    session.active_connections.add(websocket)

    # Send initial snapshot immediately
    try:
        initial_payload = session.build_payload()
        await websocket.send_text(json.dumps(initial_payload))
    except Exception as e:
        logger.error(f"Failed to send initial payload: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for {symbol}")
    except Exception as e:
        logger.warning(f"WebSocket connection error for {symbol}: {e}")
    finally:
        session.active_connections.discard(websocket)
        await asyncio.sleep(15)
        await stream_manager.cleanup_session_if_idle(symbol)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload="PORT" not in os.environ)
