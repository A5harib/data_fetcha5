/**
 * Client service to connect React/Next.js/Vite Frontend to the Python FastAPI Order Flow & ML Backend.
 */

export function connectOrderFlowAnalytics(symbol = 'BTCUSDT', onData, onError) {
  const cleanSymbol = symbol.toUpperCase().replace('-', '').replace('/', '');
  // Backend runs off-Vercel (websockets + a writable disk for retraining), so
  // dial it directly. Vercel rewrites cannot proxy a websocket upgrade.
  const base = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
  const wsUrl = `${base.replace(/^http/, 'ws').replace(/\/$/, '')}/ws/analytics/${cleanSymbol}`;
  
  let ws = null;
  let isClosedManually = false;

  function connect() {
    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log(`[OrderFlow WS] Connected to Python backend for ${cleanSymbol}`);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (onData) onData(payload);
        } catch (err) {
          console.error('[OrderFlow WS] Parse error:', err);
        }
      };

      ws.onerror = (err) => {
        if (onError) onError(err);
      };

      ws.onclose = () => {
        if (!isClosedManually) {
          console.warn('[OrderFlow WS] Disconnected. Reconnecting in 3s...');
          setTimeout(connect, 3000);
        }
      };
    } catch (e) {
      console.error('[OrderFlow WS] Setup error:', e);
    }
  }

  connect();

  return {
    disconnect: () => {
      isClosedManually = true;
      if (ws) {
        ws.close();
        ws = null;
      }
    }
  };
}
