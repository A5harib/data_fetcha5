import React, { useState } from 'react';
import { 
  Search, 
  SlidersHorizontal, 
  CandlestickChart, 
  LineChart, 
  AreaChart, 
  ChevronDown, 
  X,
  Activity
} from 'lucide-react';
import { BINANCE_MARKETS, TIMEFRAMES } from '../../services/binanceService';

export default function TerminalHeader({
  activeSymbol,
  onSelectSymbol,
  timeframe,
  onChangeTimeframe,
  chartType,
  onChangeChartType,
  indicators,
  onToggleIndicator,
  currentPrice,
  priceChange,
  priceChangePct,
  stats24h
}) {
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showIndicatorsModal, setShowIndicatorsModal] = useState(false);

  const filteredSymbols = BINANCE_MARKETS.filter(s => 
    s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) || 
    s.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleCustomSymbolSubmit = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSelectSymbol(searchQuery.trim().toUpperCase());
      setShowSearchModal(false);
      setSearchQuery('');
    }
  };

  const isUp = (priceChange || 0) >= 0;

  return (
    <header className="h-12 bg-[#181a20] border-b border-[#202630] flex items-center justify-between px-3 text-[#eaecef] select-none shrink-0 z-40">
      {/* Left: Symbol & Live Binance Ticker Stats */}
      <div className="flex items-center gap-4">
        {/* Symbol Dropdown Trigger */}
        <button 
          onClick={() => setShowSearchModal(true)}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-[#202630] hover:bg-[#2b313a] border border-[#2b313a] rounded transition-colors"
          title="Search all Binance markets"
        >
          <span className="font-bold text-sm text-[#eaecef]">{activeSymbol}</span>
          <ChevronDown size={14} className="text-[#848e9c]" />
        </button>

        {/* Live Price & Change */}
        <div className="flex items-baseline gap-2">
          <span className={`text-[17px] font-bold tabular-nums ${isUp ? 'text-[#0ECB81]' : 'text-[#F6465D]'}`}>
            ${currentPrice ? Number(currentPrice).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'}
          </span>
          <span className={`text-[11px] font-semibold px-1.5 py-0.5 rounded tabular-nums ${isUp ? 'bg-[#0ECB81]/15 text-[#0ECB81]' : 'bg-[#F6465D]/15 text-[#F6465D]'}`}>
            {isUp ? '+' : ''}{priceChange ? Number(priceChange).toFixed(2) : '0.00'} ({isUp ? '+' : ''}{priceChangePct ? Number(priceChangePct).toFixed(2) : '0.00'}%)
          </span>
        </div>

        {/* 24h Stats High / Low / Volume */}
        {stats24h && (
          <div className="hidden xl:flex items-center gap-4 text-[11px] text-[#848e9c]">
            <div className="flex flex-col">
              <span className="text-[10px] text-[#5e6673]">24h High</span>
              <span className="text-[#eaecef] font-medium font-mono">${stats24h.highPrice?.toFixed(2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-[#5e6673]">24h Low</span>
              <span className="text-[#eaecef] font-medium font-mono">${stats24h.lowPrice?.toFixed(2)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-[#5e6673]">24h Volume ({activeSymbol.replace('USDT','')})</span>
              <span className="text-[#eaecef] font-medium font-mono">{stats24h.volume?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
          </div>
        )}

        {/* Live WebSocket Status Badge */}
        <div className="hidden lg:flex items-center gap-1.5 px-2 py-0.5 bg-[#202630] border border-[#2b313a] rounded text-[11px]">
          <span className="w-2 h-2 rounded-full bg-[#0ecb81] animate-pulse" />
          <span className="text-[#0ecb81] font-semibold">Binance 24/7 WS (0ms)</span>
        </div>
      </div>

      {/* Center: Timeframe Selector & Chart Tools */}
      <div className="flex items-center gap-1 bg-[#202630] border border-[#2b313a] rounded p-0.5">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.label}
            className={`px-2.5 py-1 text-xs font-semibold rounded transition-colors ${timeframe === tf.label ? 'bg-[#2b313a] text-[#fcd535]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
            onClick={() => onChangeTimeframe(tf.label)}
          >
            {tf.label}
          </button>
        ))}

        <div className="w-[1px] h-3.5 bg-[#2b313a] mx-0.5" />

        {/* Chart Style Toggle */}
        <div className="flex items-center gap-0.5">
          <button 
            className={`p-1 rounded transition-colors ${chartType === 'candlestick' ? 'bg-[#2b313a] text-[#fcd535]' : 'text-[#848e9c] hover:text-[#eaecef]'}`} 
            onClick={() => onChangeChartType('candlestick')}
            title="Candlestick Chart"
          >
            <CandlestickChart size={15} />
          </button>
          <button 
            className={`p-1 rounded transition-colors ${chartType === 'line' ? 'bg-[#2b313a] text-[#fcd535]' : 'text-[#848e9c] hover:text-[#eaecef]'}`} 
            onClick={() => onChangeChartType('line')}
            title="Line Chart"
          >
            <LineChart size={15} />
          </button>
          <button 
            className={`p-1 rounded transition-colors ${chartType === 'area' ? 'bg-[#2b313a] text-[#fcd535]' : 'text-[#848e9c] hover:text-[#eaecef]'}`} 
            onClick={() => onChangeChartType('area')}
            title="Area Chart"
          >
            <AreaChart size={15} />
          </button>
        </div>

        <div className="w-[1px] h-3.5 bg-[#2b313a] mx-0.5" />

        {/* Indicators Trigger */}
        <button 
          className={`flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded transition-colors ${Object.values(indicators).some(Boolean) ? 'bg-[#2b313a] text-[#fcd535]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
          onClick={() => setShowIndicatorsModal(!showIndicatorsModal)}
          title="Technical Indicators"
        >
          <SlidersHorizontal size={13} />
          <span>Indicators</span>
        </button>
      </div>

      {/* Right: Exchange Badge */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 px-3 py-1 bg-[#202630] border border-[#2b313a] rounded text-xs text-[#eaecef]">
          <Activity size={13} className="text-[#fcd535]" />
          <span className="font-bold text-[#fcd535]">Binance Spot</span>
        </div>
      </div>

      {/* Modal: Symbol Search */}
      {showSearchModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setShowSearchModal(false)}>
          <div className="w-full max-w-md bg-[#1e2329] border border-[#2b313a] rounded-lg shadow-2xl overflow-hidden flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-3.5 border-b border-[#2b313a]">
              <span className="text-sm font-semibold text-[#eaecef]">Search Binance Markets</span>
              <button onClick={() => setShowSearchModal(false)} className="text-[#848e9c] hover:text-[#eaecef]">
                <X size={16} />
              </button>
            </div>

            <form onSubmit={handleCustomSymbolSubmit} className="p-3 border-b border-[#2b313a]">
              <div className="relative">
                <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#848e9c]" />
                <input 
                  type="text" 
                  className="w-full bg-[#202630] border border-[#2b313a] focus:border-[#fcd535] rounded px-3 pl-9 py-2 text-xs text-[#eaecef] outline-none placeholder-[#5e6673]" 
                  placeholder="Search BTC, ETH, SOL, XAU (Gold Perp), PAXG, DOGE..." 
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  autoFocus
                />
              </div>
            </form>

            <div className="max-h-80 overflow-y-auto divide-y divide-[#202630]">
              {filteredSymbols.map(stock => (
                <div 
                  key={stock.symbol} 
                  className={`flex justify-between items-center px-4 py-2.5 hover:bg-[#2b313a] cursor-pointer transition-colors ${activeSymbol === stock.symbol ? 'bg-[#2b313a]' : ''}`}
                  onClick={() => {
                    onSelectSymbol(stock.symbol);
                    setShowSearchModal(false);
                    setSearchQuery('');
                  }}
                >
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#eaecef] flex items-center gap-1.5">
                      {stock.symbol}
                      <span className="text-[10px] bg-[#0ecb81]/15 text-[#0ecb81] px-1 rounded font-normal">
                        24/7 WS
                      </span>
                    </span>
                    <span className="text-[11px] text-[#848e9c]">{stock.name}</span>
                  </div>
                  <span className="text-[11px] text-[#848e9c] bg-[#202630] px-2 py-0.5 rounded border border-[#2b313a]">
                    {stock.sector}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Modal: Full Technical Indicators Suite */}
      {showIndicatorsModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setShowIndicatorsModal(false)}>
          <div className="w-full max-w-md bg-[#1e2329] border border-[#2b313a] rounded-lg shadow-2xl overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-3.5 border-b border-[#2b313a]">
              <span className="text-sm font-semibold text-[#eaecef]">Technical Indicators</span>
              <button onClick={() => setShowIndicatorsModal(false)} className="text-[#848e9c] hover:text-[#eaecef]">
                <X size={16} />
              </button>
            </div>

            <div className="p-4 grid grid-cols-1 gap-2.5">
              {/* VWAP */}
              <label className="flex items-center justify-between p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer hover:border-[#474d57]">
                <div className="flex items-center gap-2.5">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.vwap} 
                    onChange={() => onToggleIndicator('vwap')} 
                    className="accent-[#eab308] w-4 h-4 rounded cursor-pointer"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#eab308]">VWAP (Institutional Benchmark)</span>
                    <span className="text-[10px] text-[#848e9c]">Volume-Weighted Average Price</span>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#eab308] bg-[#eab308]/10 px-1.5 py-0.5 rounded">Overlay</span>
              </label>

              {/* RSI (14) */}
              <label className="flex items-center justify-between p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer hover:border-[#474d57]">
                <div className="flex items-center gap-2.5">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.rsi} 
                    onChange={() => onToggleIndicator('rsi')} 
                    className="accent-[#c084fc] w-4 h-4 rounded cursor-pointer"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#c084fc]">RSI 14 (Overbought/Oversold)</span>
                    <span className="text-[10px] text-[#848e9c]">Relative Strength Index with 70/30 levels</span>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#c084fc] bg-[#c084fc]/10 px-1.5 py-0.5 rounded">Oscillator</span>
              </label>

              {/* MACD */}
              <label className="flex items-center justify-between p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer hover:border-[#474d57]">
                <div className="flex items-center gap-2.5">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.macd} 
                    onChange={() => onToggleIndicator('macd')} 
                    className="accent-[#38bdf8] w-4 h-4 rounded cursor-pointer"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#38bdf8]">MACD (12, 26, 9)</span>
                    <span className="text-[10px] text-[#848e9c]">Momentum histogram & signal line</span>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#38bdf8] bg-[#38bdf8]/10 px-1.5 py-0.5 rounded">Oscillator</span>
              </label>

              {/* Bollinger Bands */}
              <label className="flex items-center justify-between p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer hover:border-[#474d57]">
                <div className="flex items-center gap-2.5">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.bollinger} 
                    onChange={() => onToggleIndicator('bollinger')} 
                    className="accent-[#38bdf8] w-4 h-4 rounded cursor-pointer"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#38bdf8]">Bollinger Bands (20, 2)</span>
                    <span className="text-[10px] text-[#848e9c]">Volatility expansion & squeeze envelope</span>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#38bdf8] bg-[#38bdf8]/10 px-1.5 py-0.5 rounded">Envelope</span>
              </label>

              {/* SMA 200 */}
              <label className="flex items-center justify-between p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer hover:border-[#474d57]">
                <div className="flex items-center gap-2.5">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.sma200} 
                    onChange={() => onToggleIndicator('sma200')} 
                    className="accent-[#a855f7] w-4 h-4 rounded cursor-pointer"
                  />
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-[#a855f7]">SMA 200 (Macro Bull/Bear Line)</span>
                    <span className="text-[10px] text-[#848e9c]">200-period institutional moving average</span>
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[#a855f7] bg-[#a855f7]/10 px-1.5 py-0.5 rounded">Macro MA</span>
              </label>

              {/* EMA 9 / 21 Ribbon */}
              <div className="grid grid-cols-2 gap-2">
                <label className="flex items-center gap-2 p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.ema9} 
                    onChange={() => onToggleIndicator('ema9')} 
                    className="accent-[#818cf8] w-4 h-4 rounded cursor-pointer"
                  />
                  <span className="text-xs font-bold text-[#818cf8]">EMA 9</span>
                </label>
                <label className="flex items-center gap-2 p-2 rounded bg-[#202630] border border-[#2b313a] cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={!!indicators.ema21} 
                    onChange={() => onToggleIndicator('ema21')} 
                    className="accent-[#ec4899] w-4 h-4 rounded cursor-pointer"
                  />
                  <span className="text-xs font-bold text-[#ec4899]">EMA 21</span>
                </label>
              </div>
            </div>

            <div className="p-3 border-t border-[#2b313a] flex justify-end">
              <button 
                onClick={() => setShowIndicatorsModal(false)}
                className="bg-[#fcd535] hover:bg-[#e0be30] text-[#181a20] font-bold text-xs px-4 py-1.5 rounded transition-colors"
              >
                Apply Indicators
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
