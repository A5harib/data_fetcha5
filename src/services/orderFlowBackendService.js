/**
 * Client service to connect React/Next.js/Vite Frontend to the Python FastAPI Order Flow & ML Backend.
 */

export function connectOrderFlowAnalytics(symbol = 'BTCUSDT', onData, onError) {
  const cleanSymbol = symbol.toUpperCase().replace('-', '').replace('/', '');
  // ponytail: same-origin in prod (vercel.json rewrites /ws to the backend service), localhost in dev
  const host = import.meta.env.DEV ? 'localhost:8000' : window.location.host;
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${host}/ws/analytics/${cleanSymbol}`;
  
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
