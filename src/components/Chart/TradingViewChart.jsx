import React, { useEffect, useRef, useState } from 'react';
import { 
  createChart, 
  CandlestickSeries, 
  LineSeries, 
  AreaSeries, 
  HistogramSeries, 
  CrosshairMode 
} from 'lightweight-charts';
import { 
  calculateSMA, 
  calculateEMA, 
  calculateVWAP,
  calculateBollingerBands,
  calculateRSI,
  calculateMACD
} from '../../utils/indicators';
import { 
  RotateCcw, 
  Camera
} from 'lucide-react';

export default function TradingViewChart({
  symbol,
  timeframe,
  chartType = 'candlestick',
  data = [],
  volumeData = [],
  indicators = {},
  liveKlineUpdate = null
}) {
  const chartContainerRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const mainSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const indicatorSeriesRefs = useRef({});

  const [legendData, setLegendData] = useState(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const container = chartContainerRef.current;

    const hasOscillators = indicators.rsi || indicators.macd;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight || 560,
      layout: {
        background: { color: '#12161f' },
        textColor: '#848e9c',
        fontSize: 11,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif"
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.03)' }
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: 'rgba(255, 255, 255, 0.2)',
          width: 1,
          style: 3,
          labelBackgroundColor: '#2b313a'
        },
        horzLine: {
          color: 'rgba(255, 255, 255, 0.2)',
          width: 1,
          style: 3,
          labelBackgroundColor: '#2b313a'
        }
      },
      rightPriceScale: {
        borderColor: '#202630',
        scaleMargins: { 
          top: 0.05, 
          bottom: hasOscillators ? 0.32 : 0.18 
        },
        autoScale: true
      },
      timeScale: {
        borderColor: '#202630',
        timeVisible: true,
        secondsVisible: false,
        barSpacing: 9,
        minBarSpacing: 3
      },
      watermark: {
        visible: true,
        fontSize: 52,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(255, 255, 255, 0.02)',
        text: `${symbol} · ${timeframe}`
      }
    });

    chartInstanceRef.current = chart;

    const addSeriesHelper = (SeriesClass, options) => {
      if (chart.addSeries) {
        return chart.addSeries(SeriesClass, options);
      }
      return null;
    };

    let mainSeries;
    if (chartType === 'line') {
      mainSeries = addSeriesHelper(LineSeries, {
        color: '#fcd535',
        lineWidth: 2,
        crosshairMarkerVisible: true
      });
    } else if (chartType === 'area') {
      mainSeries = addSeriesHelper(AreaSeries, {
        topColor: 'rgba(252, 213, 53, 0.3)',
        bottomColor: 'rgba(252, 213, 53, 0.0)',
        lineColor: '#fcd535',
        lineWidth: 2
      });
    } else {
      mainSeries = addSeriesHelper(CandlestickSeries, {
        upColor: '#0ECB81',
        downColor: '#F6465D',
        borderVisible: false,
        wickUpColor: '#0ECB81',
        wickDownColor: '#F6465D'
      });
    }
    mainSeriesRef.current = mainSeries;

    // Volume Series
    const volumeSeries = addSeriesHelper(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume_scale'
    });
    volumeSeriesRef.current = volumeSeries;

    chart.priceScale('volume_scale').applyOptions({
      scaleMargins: { top: hasOscillators ? 0.65 : 0.82, bottom: hasOscillators ? 0.28 : 0 }
    });

    if (data && data.length > 0) {
      if (chartType === 'line' || chartType === 'area') {
        mainSeries.setData(data.map(d => ({ time: d.time, value: d.close })));
      } else {
        mainSeries.setData(data);
      }

      if (volumeData && volumeData.length > 0) {
        volumeSeries.setData(volumeData);
      }

      const last = data[data.length - 1];
      const prev = data.length > 1 ? data[data.length - 2] : last;
      const chg = last.close - prev.close;
      const chgPct = (chg / prev.close) * 100;
      setLegendData({
        open: last.open,
        high: last.high,
        low: last.low,
        close: last.close,
        change: chg,
        changePct: chgPct,
        volume: volumeData[volumeData.length - 1]?.value || 0
      });
    }

    // Technical Indicators
    const indRefs = {};

    // 1. VWAP (Volume-Weighted Average Price - Institutional)
    if (indicators.vwap && data.length > 0) {
      const vwapSeries = addSeriesHelper(LineSeries, { 
        color: '#eab308', 
        lineWidth: 2, 
        lineStyle: 1, // Dotted
        title: 'VWAP' 
      });
      vwapSeries.setData(calculateVWAP(data, volumeData));
      indRefs.vwap = vwapSeries;
    }

    // 2. SMA 20
    if (indicators.sma20 && data.length >= 20) {
      const smaSeries = addSeriesHelper(LineSeries, { color: '#06b6d4', lineWidth: 1.5, title: 'SMA 20' });
      smaSeries.setData(calculateSMA(data, 20));
      indRefs.sma20 = smaSeries;
    }

    // 3. SMA 50
    if (indicators.sma50 && data.length >= 50) {
      const smaSeries = addSeriesHelper(LineSeries, { color: '#f59e0b', lineWidth: 1.5, title: 'SMA 50' });
      smaSeries.setData(calculateSMA(data, 50));
      indRefs.sma50 = smaSeries;
    }

    // 4. SMA 200 (Wall Street Bull/Bear Line)
    if (indicators.sma200 && data.length >= 200) {
      const smaSeries = addSeriesHelper(LineSeries, { color: '#a855f7', lineWidth: 2, title: 'SMA 200' });
      smaSeries.setData(calculateSMA(data, 200));
      indRefs.sma200 = smaSeries;
    }

    // 5. EMA 9
    if (indicators.ema9 && data.length >= 9) {
      const emaSeries = addSeriesHelper(LineSeries, { color: '#818cf8', lineWidth: 1.5, title: 'EMA 9' });
      emaSeries.setData(calculateEMA(data, 9));
      indRefs.ema9 = emaSeries;
    }

    // 6. EMA 21
    if (indicators.ema21 && data.length >= 21) {
      const emaSeries = addSeriesHelper(LineSeries, { color: '#ec4899', lineWidth: 1.5, title: 'EMA 21' });
      emaSeries.setData(calculateEMA(data, 21));
      indRefs.ema21 = emaSeries;
    }

    // 7. Bollinger Bands
    if (indicators.bollinger && data.length >= 20) {
      const bb = calculateBollingerBands(data, 20, 2);
      const upperSeries = addSeriesHelper(LineSeries, { color: 'rgba(56, 189, 248, 0.7)', lineWidth: 1, lineStyle: 2 });
      const midSeries = addSeriesHelper(LineSeries, { color: 'rgba(148, 163, 184, 0.6)', lineWidth: 1, lineStyle: 0 });
      const lowerSeries = addSeriesHelper(LineSeries, { color: 'rgba(56, 189, 248, 0.7)', lineWidth: 1, lineStyle: 2 });

      upperSeries.setData(bb.upper);
      midSeries.setData(bb.middle);
      lowerSeries.setData(bb.lower);

      indRefs.bbUpper = upperSeries;
      indRefs.bbMid = midSeries;
      indRefs.bbLower = lowerSeries;
    }

    // 8. RSI (14) - Oscillator Sub-scale
    if (indicators.rsi && data.length >= 14) {
      const rsiSeries = addSeriesHelper(LineSeries, {
        color: '#c084fc',
        lineWidth: 2,
        priceScaleId: 'rsi_scale',
        title: 'RSI 14'
      });
      rsiSeries.setData(calculateRSI(data, 14));

      // Overbought 70 line
      const obLine = addSeriesHelper(LineSeries, {
        color: 'rgba(246, 70, 93, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'rsi_scale'
      });
      obLine.setData(data.map(d => ({ time: d.time, value: 70 })));

      // Oversold 30 line
      const osLine = addSeriesHelper(LineSeries, {
        color: 'rgba(14, 203, 129, 0.5)',
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'rsi_scale'
      });
      osLine.setData(data.map(d => ({ time: d.time, value: 30 })));

      chart.priceScale('rsi_scale').applyOptions({
        scaleMargins: { top: 0.78, bottom: 0.02 },
        autoScale: false
      });

      indRefs.rsi = rsiSeries;
    }

    // 9. MACD (12, 26, 9)
    if (indicators.macd && data.length >= 35) {
      const macd = calculateMACD(data);
      const macdLineSeries = addSeriesHelper(LineSeries, {
        color: '#38bdf8',
        lineWidth: 1.5,
        priceScaleId: 'macd_scale',
        title: 'MACD'
      });
      const signalLineSeries = addSeriesHelper(LineSeries, {
        color: '#f97316',
        lineWidth: 1.5,
        priceScaleId: 'macd_scale',
        title: 'Signal'
      });
      const histSeries = addSeriesHelper(HistogramSeries, {
        priceScaleId: 'macd_scale'
      });

      macdLineSeries.setData(macd.macdLine);
      signalLineSeries.setData(macd.signalLine);
      histSeries.setData(macd.histogram);

      chart.priceScale('macd_scale').applyOptions({
        scaleMargins: { top: 0.80, bottom: 0.02 }
      });

      indRefs.macdLine = macdLineSeries;
      indRefs.macdSignal = signalLineSeries;
      indRefs.macdHist = histSeries;
    }

    indicatorSeriesRefs.current = indRefs;

    // Crosshair Move
    chart.subscribeCrosshairMove((param) => {
      if (!param || !param.time || !param.seriesData) {
        if (data && data.length > 0) {
          const last = data[data.length - 1];
          const prev = data.length > 1 ? data[data.length - 2] : last;
          setLegendData({
            open: last.open,
            high: last.high,
            low: last.low,
            close: last.close,
            change: last.close - prev.close,
            changePct: ((last.close - prev.close) / prev.close) * 100,
            volume: volumeData[volumeData.length - 1]?.value || 0
          });
        }
        return;
      }

      const barData = param.seriesData.get(mainSeries);
      const volBar = param.seriesData.get(volumeSeries);

      if (barData) {
        const o = barData.open !== undefined ? barData.open : barData.value;
        const h = barData.high !== undefined ? barData.high : barData.value;
        const l = barData.low !== undefined ? barData.low : barData.value;
        const c = barData.close !== undefined ? barData.close : barData.value;
        const chg = c - o;
        const chgPct = (chg / (o || 1)) * 100;

        setLegendData({
          open: o,
          high: h,
          low: l,
          close: c,
          change: chg,
          changePct: chgPct,
          volume: volBar?.value || 0
        });
      }
    });

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight || 560
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
        chartInstanceRef.current = null;
      }
    };
  }, [symbol, timeframe, chartType, indicators, data]);

  // Real-Time Live Binance WebSocket Kline Streaming
  useEffect(() => {
    if (!liveKlineUpdate || !mainSeriesRef.current || !data || data.length === 0) return;

    const lastBar = data[data.length - 1];
    // Guard against out-of-order ticks or stale timeframe ticks
    if (lastBar && liveKlineUpdate.time < lastBar.time) {
      return;
    }

    try {
      if (chartType === 'line' || chartType === 'area') {
        mainSeriesRef.current.update({
          time: liveKlineUpdate.time,
          value: liveKlineUpdate.close
        });
      } else {
        mainSeriesRef.current.update({
          time: liveKlineUpdate.time,
          open: liveKlineUpdate.open,
          high: liveKlineUpdate.high,
          low: liveKlineUpdate.low,
          close: liveKlineUpdate.close
        });
      }

      if (volumeSeriesRef.current && liveKlineUpdate.volume !== undefined) {
        volumeSeriesRef.current.update({
          time: liveKlineUpdate.time,
          value: liveKlineUpdate.volume,
          color: liveKlineUpdate.close >= liveKlineUpdate.open ? 'rgba(14, 203, 129, 0.45)' : 'rgba(246, 70, 93, 0.45)'
        });
      }

      setLegendData(prev => {
        if (!prev) return prev;
        const chg = liveKlineUpdate.close - liveKlineUpdate.open;
        const chgPct = (chg / (liveKlineUpdate.open || 1)) * 100;
        return {
          ...prev,
          open: liveKlineUpdate.open,
          high: Math.max(prev.high || liveKlineUpdate.high, liveKlineUpdate.high),
          low: Math.min(prev.low || liveKlineUpdate.low, liveKlineUpdate.low),
          close: liveKlineUpdate.close,
          change: chg,
          changePct: chgPct,
          volume: liveKlineUpdate.volume
        };
      });
    } catch (err) {
      // Silently ignore transient race conditions during rapid timeframe switching
      console.debug('[TradingViewChart] Live tick sync skipped during timeframe transition:', err.message);
    }
  }, [liveKlineUpdate, chartType, data]);

  const handleFitContent = () => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.timeScale().fitContent();
    }
  };

  const handleTakeScreenshot = () => {
    if (chartInstanceRef.current) {
      const canvas = chartInstanceRef.current.takeScreenshot();
      const link = document.createElement('a');
      link.download = `${symbol}_${timeframe}_chart.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };

  return (
    <div className="flex-1 flex flex-col relative bg-[#12161f] min-h-0 overflow-hidden">
      {/* Top Legend Bar */}
      <div className="h-8 flex items-center justify-between px-3 bg-[#181a20]/80 border-b border-[#202630] select-none shrink-0 z-10">
        {legendData && (
          <div className="flex items-center gap-3 text-[11px] font-mono text-[#848e9c]">
            <span className="font-bold text-[#eaecef]">{symbol}</span>
            <span className="bg-[#202630] text-[#fcd535] px-1.5 py-0.5 rounded text-[10px]">{timeframe}</span>
            <span>O <strong className="text-[#eaecef]">{legendData.open?.toFixed(2)}</strong></span>
            <span>H <strong className="text-[#eaecef]">{legendData.high?.toFixed(2)}</strong></span>
            <span>L <strong className="text-[#eaecef]">{legendData.low?.toFixed(2)}</strong></span>
            <span>C <strong className={legendData.change >= 0 ? 'text-[#0ECB81]' : 'text-[#F6465D]'}>{legendData.close?.toFixed(2)}</strong></span>
            <span className={legendData.change >= 0 ? 'text-[#0ECB81]' : 'text-[#F6465D]'}>
              {legendData.change >= 0 ? '+' : ''}{legendData.change?.toFixed(2)} ({legendData.changePct >= 0 ? '+' : ''}{legendData.changePct?.toFixed(2)}%)
            </span>
            <span>Vol <strong className="text-[#eaecef]">{legendData.volume?.toLocaleString()}</strong></span>
          </div>
        )}

        <div className="flex items-center gap-1.5">
          <button 
            onClick={handleFitContent}
            className="flex items-center gap-1 px-2 py-0.5 bg-[#202630] hover:bg-[#2b313a] border border-[#2b313a] rounded text-[10px] text-[#848e9c] hover:text-[#eaecef]"
            title="Reset Zoom"
          >
            <RotateCcw size={11} />
            <span>Fit</span>
          </button>
          <button 
            onClick={handleTakeScreenshot}
            className="flex items-center gap-1 px-2 py-0.5 bg-[#202630] hover:bg-[#2b313a] border border-[#2b313a] rounded text-[10px] text-[#848e9c] hover:text-[#eaecef]"
            title="Export PNG"
          >
            <Camera size={11} />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div 
        ref={chartContainerRef} 
        className="flex-1 w-full h-full relative"
      />
    </div>
  );
}
