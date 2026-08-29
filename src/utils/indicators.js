/**
 * Technical Indicator Calculation Utilities for Financial Charts
 */

// Simple Moving Average (SMA)
export function calculateSMA(data, period = 20) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    result.push({
      time: data[i].time,
      value: Number((sum / period).toFixed(2))
    });
  }
  return result;
}

// Exponential Moving Average (EMA)
export function calculateEMA(data, period = 9) {
  const result = [];
  if (data.length < period) return result;

  const k = 2 / (period + 1);
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += data[i].close;
  }
  let ema = sum / period;
  result.push({ time: data[period - 1].time, value: Number(ema.toFixed(2)) });

  for (let i = period; i < data.length; i++) {
    ema = data[i].close * k + ema * (1 - k);
    result.push({
      time: data[i].time,
      value: Number(ema.toFixed(2))
    });
  }
  return result;
}

// Volume-Weighted Average Price (VWAP)
export function calculateVWAP(data, volumeData = []) {
  const result = [];
  let cumulativeTypicalVol = 0;
  let cumulativeVol = 0;

  for (let i = 0; i < data.length; i++) {
    const bar = data[i];
    const vol = volumeData[i]?.value || bar.volume || 1;
    const typicalPrice = (bar.high + bar.low + bar.close) / 3;

    cumulativeTypicalVol += typicalPrice * vol;
    cumulativeVol += vol;

    const vwap = cumulativeVol > 0 ? cumulativeTypicalVol / cumulativeVol : bar.close;
    result.push({
      time: bar.time,
      value: Number(vwap.toFixed(2))
    });
  }
  return result;
}

// Bollinger Bands (period = 20, multiplier = 2)
export function calculateBollingerBands(data, period = 20, multiplier = 2) {
  const upper = [];
  const middle = [];
  const lower = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) continue;

    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j].close;
    }
    const sma = sum / period;

    let varianceSum = 0;
    for (let j = 0; j < period; j++) {
      varianceSum += Math.pow(data[i - j].close - sma, 2);
    }
    const stdDev = Math.sqrt(varianceSum / period);

    upper.push({ time: data[i].time, value: Number((sma + multiplier * stdDev).toFixed(2)) });
    middle.push({ time: data[i].time, value: Number(sma.toFixed(2)) });
    lower.push({ time: data[i].time, value: Number((sma - multiplier * stdDev).toFixed(2)) });
  }

  return { upper, middle, lower };
}

// Relative Strength Index (RSI, 14-period)
export function calculateRSI(data, period = 14) {
  const result = [];
  if (data.length <= period) return result;

  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const change = data[i].close - data[i - 1].close;
    if (change >= 0) gains += change;
    else losses += Math.abs(change);
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  let rsi = 100 - (100 / (1 + rs));
  result.push({ time: data[period].time, value: Number(rsi.toFixed(2)) });

  for (let i = period + 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close;
    const currentGain = change > 0 ? change : 0;
    const currentLoss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + currentGain) / period;
    avgLoss = (avgLoss * (period - 1) + currentLoss) / period;

    rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi = 100 - (100 / (1 + rs));

    result.push({
      time: data[i].time,
      value: Number(rsi.toFixed(2))
    });
  }

  return result;
}

// Moving Average Convergence Divergence (MACD: 12, 26, 9)
export function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (data.length < slowPeriod + signalPeriod) {
    return { macdLine: [], signalLine: [], histogram: [] };
  }

  const fastEma = calculateEMA(data, fastPeriod);
  const slowEma = calculateEMA(data, slowPeriod);

  const slowTimeMap = new Map();
  slowEma.forEach(item => slowTimeMap.set(item.time, item.value));

  const macdData = [];
  fastEma.forEach(item => {
    if (slowTimeMap.has(item.time)) {
      const slowVal = slowTimeMap.get(item.time);
      const macdVal = Number((item.value - slowVal).toFixed(2));
      macdData.push({ time: item.time, close: macdVal, value: macdVal });
    }
  });

  const signalEma = calculateEMA(macdData, signalPeriod);
  const signalMap = new Map();
  signalEma.forEach(item => signalMap.set(item.time, item.value));

  const macdLine = [];
  const signalLine = [];
  const histogram = [];

  macdData.forEach(item => {
    if (signalMap.has(item.time)) {
      const sVal = signalMap.get(item.time);
      const histVal = Number((item.value - sVal).toFixed(2));

      macdLine.push({ time: item.time, value: item.value });
      signalLine.push({ time: item.time, value: sVal });
      histogram.push({
        time: item.time,
        value: histVal,
        color: histVal >= 0 ? '#0ECB81' : '#F6465D'
      });
    }
  });

  return { macdLine, signalLine, histogram };
}
