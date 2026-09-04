/**
 * 100% Pure 24/7 Binance WebSocket & REST API Engine
 * Zero delays, zero synthetic data, zero Polygon dependency.
 */

export const BINANCE_MARKETS = [
  { symbol: 'BTCUSDT', name: 'Bitcoin', sector: 'Layer 1' },
  { symbol: 'ETHUSDT', name: 'Ethereum', sector: 'Smart Contracts' },
  { symbol: 'SOLUSDT', name: 'Solana', sector: 'High-speed L1' },
  { symbol: 'XAUUSDT', name: 'Gold (XAU Perp)', sector: 'Precious Metals' },
  { symbol: 'PAXGUSDT', name: 'PAX Gold (PAXG)', sector: 'Precious Metals' },
  { symbol: 'BNBUSDT', name: 'BNB', sector: 'Ecosystem' },
  { symbol: 'XRPUSDT', name: 'XRP', sector: 'Payments' },
  { symbol: 'DOGEUSDT', name: 'Dogecoin', sector: 'Meme' },
  { symbol: 'ADAUSDT', name: 'Cardano', sector: 'Layer 1' },
  { symbol: 'AVAXUSDT', name: 'Avalanche', sector: 'Layer 1' },
  { symbol: 'LINKUSDT', name: 'Chainlink', sector: 'Oracle' },
  { symbol: 'SUIUSDT', name: 'Sui Network', sector: 'Layer 1' },
  { symbol: 'NEARUSDT', name: 'NEAR Protocol', sector: 'Layer 1' }
];

export const TIMEFRAMES = [
  { label: '1s', interval: '1s' },
  { label: '1m', interval: '1m' },
  { label: '5m', interval: '5m' },
  { label: '15m', interval: '15m' },
  { label: '1h', interval: '1h' },
  { label: '4h', interval: '4h' },
  { label: '1D', interval: '1d' },
  { label: '1W', interval: '1w' }
];

export function normalizeBinanceSymbol(raw) {
  if (!raw) return 'BTCUSDT';
  const clean = raw.trim().toUpperCase().replace('X:', '').replace('C:', '').replace('-', '').replace('/', '');
  if (clean === 'XAU' || clean === 'XAUUSD' || clean === 'GOLD') return 'XAUUSDT';
  if (clean === 'PAXG') return 'PAXGUSDT';
  if (clean === 'BTC' || clean === 'BTCUSD') return 'BTCUSDT';
  if (clean === 'ETH' || clean === 'ETHUSD') return 'ETHUSDT';
  if (clean === 'SOL' || clean === 'SOLUSD') return 'SOLUSDT';
  if (clean.endsWith('USDT')) return clean;
  if (clean.endsWith('USD')) return clean + 'T';
  return `${clean}USDT`;
}

// ponytail: XAU/XAG are USDS-M TradFi perps, not spot listings. Only the host differs,
// so route by symbol instead of forking the service. Add more symbols here if Binance
// lists further TradFi perps.
const FUTURES_ONLY = new Set(['XAUUSDT', 'XAGUSDT']);

export function isFuturesSymbol(symbol) {
  return FUTURES_ONLY.has(symbol);
}

function restBase(symbol) {
  return isFuturesSymbol(symbol)
    ? 'https://fapi.binance.com/fapi/v1'
    : 'https://api.binance.com/api/v3';
}

/**
 * Fetch Historical Candlesticks from Binance Spot REST API
 */
export async function fetchBinanceKlines(rawSymbol, timeframe = '1D') {
  const symbol = normalizeBinanceSymbol(rawSymbol);
  const tfObj = TIMEFRAMES.find(t => t.label === timeframe) || { interval: '1d' };
  const interval = tfObj.interval;
  const limit = (interval === '1s' || interval === '1m' || interval === '5m' || interval === '15m') ? 350 : 500;

  const url = `${restBase(symbol)}/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Binance API Error: HTTP ${res.status} ${res.statusText}`);
  }

  const data = await res.json();
  const candles = [];
  const volumes = [];
  const rawBars = [];

  data.forEach((bar) => {
    const time = Math.floor(bar[0] / 1000);
    const open = Number(parseFloat(bar[1]).toFixed(2));
    const high = Number(parseFloat(bar[2]).toFixed(2));
    const low = Number(parseFloat(bar[3]).toFixed(2));
    const close = Number(parseFloat(bar[4]).toFixed(2));
    const volume = parseFloat(bar[5]);
    const isUp = close >= open;

    candles.push({ time, open, high, low, close });
    volumes.push({
      time,
      value: volume,
      color: isUp ? 'rgba(14, 203, 129, 0.45)' : 'rgba(246, 70, 93, 0.45)'
    });

    rawBars.push({
      timestamp: new Date(bar[0]).toLocaleString(),
      open,
      high,
      low,
      close,
      volume,
      vwap: close,
      trades: bar[8] || 0
    });
  });

  const lastBar = candles[candles.length - 1] || { close: 0 };
  const prevBar = candles.length > 1 ? candles[candles.length - 2] : lastBar;

  const priceChange = lastBar.close - prevBar.close;
  const priceChangePct = prevBar.close > 0 ? (priceChange / prevBar.close) * 100 : 0;

  return {
    symbol,
    candles,
    volumes,
    rawBars: rawBars.reverse(),
    lastPrice: lastBar.close,
    priceChange,
    priceChangePct,
    totalCount: candles.length
  };
}

/**
 * Fetch 24hr Ticker Statistics
 */
export async function fetchBinance24hStats(rawSymbol) {
  const symbol = normalizeBinanceSymbol(rawSymbol);
  const res = await fetch(`${restBase(symbol)}/ticker/24hr?symbol=${symbol}`);
  if (!res.ok) return null;
  const data = await res.json();
  return {
    symbol,
    priceChange: parseFloat(data.priceChange),
    priceChangePercent: parseFloat(data.priceChangePercent),
    lastPrice: parseFloat(data.lastPrice),
    highPrice: parseFloat(data.highPrice),
    lowPrice: parseFloat(data.lowPrice),
    volume: parseFloat(data.volume),
    quoteVolume: parseFloat(data.quoteVolume),
    openTime: data.openTime,
    closeTime: data.closeTime
  };
}

/**
 * Connect to Multi-Stream 24/7 Binance WebSocket
 * Streams:
 * 1. <symbol>@kline_<interval> (Sub-second live candle tick)
 * 2. <symbol>@depth20@100ms (100ms Level 2 Order Book)
 * 3. <symbol>@trade (Live real-time trade tape)
 * 4. <symbol>@ticker (24h stats)
 */
export function connectBinanceMultiStream(rawSymbol, timeframe, { onKline, onDepth, onTrade, onTicker }) {
  const upper = normalizeBinanceSymbol(rawSymbol);
  const symbol = upper.toLowerCase();
  const tfObj = TIMEFRAMES.find(t => t.label === timeframe) || { interval: '1d' };
  const interval = tfObj.interval || '1d';

  // ponytail: futures has no @trade stream, @aggTrade is the equivalent and carries the
  // same p/q/T/m fields (id is `a` not `t`).
  const futures = isFuturesSymbol(upper);
  const wsHost = futures ? 'wss://fstream.binance.com' : 'wss://stream.binance.com:9443';
  const tradeStream = futures ? 'aggTrade' : 'trade';

  const streamUrl = `${wsHost}/stream?streams=${symbol}@kline_${interval}/${symbol}@depth20@100ms/${symbol}@${tradeStream}/${symbol}@ticker`;

  let ws = null;
  let isClosedManually = false;

  function connect() {
    ws = new WebSocket(streamUrl);

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const { stream, data } = message;
        if (!data) return;

        // Candlestick Stream
        if (stream.includes('@kline')) {
          const k = data.k;
          if (k && onKline) {
            onKline({
              time: Math.floor(k.t / 1000),
              open: Number(parseFloat(k.o).toFixed(2)),
              high: Number(parseFloat(k.h).toFixed(2)),
              low: Number(parseFloat(k.l).toFixed(2)),
              close: Number(parseFloat(k.c).toFixed(2)),
              volume: parseFloat(k.v),
              isClosed: k.x
            });
          }
        }

        // Level 2 Order Book (100ms)
        else if (stream.includes('@depth')) {
          if (onDepth && data.bids && data.asks) {
            let cumBid = 0;
            const bids = data.bids.map(([price, size]) => {
              const p = parseFloat(price);
              const s = parseFloat(size);
              cumBid += s;
              return { price: p, size: s, total: Number(cumBid.toFixed(3)) };
            });

            let cumAsk = 0;
            const asks = data.asks.map(([price, size]) => {
              const p = parseFloat(price);
              const s = parseFloat(size);
              cumAsk += s;
              return { price: p, size: s, total: Number(cumAsk.toFixed(3)) };
            });

            const spread = asks.length > 0 && bids.length > 0 ? Number((asks[0].price - bids[0].price).toFixed(2)) : 0;
            onDepth({ bids, asks, spread });
          }
        }

        // Live Real-Time Trade Stream
        else if (stream.includes('@trade') || stream.includes('@aggTrade')) {
          if (onTrade && data.p && data.q) {
            onTrade({
              id: data.t ?? data.a,
              time: new Date(data.T).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              price: parseFloat(data.p),
              size: parseFloat(data.q),
              side: data.m ? 'SELL' : 'BUY'
            });
          }
        }

        // 24h Ticker Stats
        else if (stream.includes('@ticker')) {
          if (onTicker && data.c && data.P) {
            onTicker({
              lastPrice: parseFloat(data.c),
              priceChange: parseFloat(data.p),
              priceChangePercent: parseFloat(data.P),
              high24h: parseFloat(data.h),
              low24h: parseFloat(data.l),
              volume24h: parseFloat(data.v)
            });
          }
        }
      } catch (err) {
        console.warn('[Binance WS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      if (!isClosedManually) {
        setTimeout(connect, 2000);
      }
    };
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
