import React, { useState, useMemo } from 'react';
import { MoreHorizontal, ArrowUp, ArrowDown, ChevronRight, ChevronDown } from 'lucide-react';

export default function BinanceOrderBook({
  symbol = 'BTCUSDT',
  currentPrice,
  priceChange,
  orderBook = null
}) {
  const [viewMode, setViewMode] = useState('ALL'); // 'ALL', 'BIDS', 'ASKS'
  const [precision, setPrecision] = useState(2);

  const baseSymbol = symbol.replace('USDT', '').replace('USD', '').replace('X:', '');
  const quoteSymbol = 'USDT';

  const isUp = (priceChange || 0) >= 0;

  const formatTotal = (totalAmount) => {
    if (!totalAmount) return '0.00';
    if (totalAmount >= 1000000) return `${(totalAmount / 1000000).toFixed(2)}M`;
    if (totalAmount >= 1000) return `${(totalAmount / 1000).toFixed(2)}K`;
    if (totalAmount >= 100) return totalAmount.toFixed(2);
    if (totalAmount >= 10) return totalAmount.toFixed(4);
    return totalAmount.toFixed(6);
  };

  const formatPrice = (p) => {
    return Number(p).toLocaleString('en-US', {
      minimumFractionDigits: precision,
      maximumFractionDigits: precision
    });
  };

  const formatAmount = (amt) => {
    return Number(amt).toFixed(5);
  };

  const asks = orderBook?.asks || [];
  const bids = orderBook?.bids || [];

  const rowCount = viewMode === 'ALL' ? 12 : 24;

  const displayAsks = useMemo(() => {
    if (viewMode === 'BIDS') return [];
    return asks.slice(0, rowCount).reverse();
  }, [asks, rowCount, viewMode]);

  const displayBids = useMemo(() => {
    if (viewMode === 'ASKS') return [];
    return bids.slice(0, rowCount);
  }, [bids, rowCount, viewMode]);

  const maxAskTotal = displayAsks.length > 0 ? displayAsks[0].total : 1;
  const maxBidTotal = displayBids.length > 0 ? displayBids[displayBids.length - 1].total : 1;
  const maxTotal = Math.max(maxAskTotal, maxBidTotal, 1);

  return (
    <div className="flex flex-col bg-[#181a20] text-[#eaecef] select-none h-full border-b border-[#202630]">
      {/* 1. Header */}
      <div className="flex items-center justify-between px-3.5 pt-2.5 pb-1.5">
        <span className="text-[13px] font-semibold text-[#eaecef]">Order Book</span>
        <button className="text-[#848e9c] hover:text-[#eaecef] p-0.5" title="Options">
          <MoreHorizontal size={15} />
        </button>
      </div>

      {/* 2. Controls: Mode Icons & Precision */}
      <div className="flex items-center justify-between px-3.5 pb-2">
        {/* 3 View Mode Buttons */}
        <div className="flex items-center gap-1.5">
          <button 
            className={`w-6 h-5 flex items-center justify-center rounded border transition-colors ${viewMode === 'ALL' ? 'border-[#fcd535] bg-[#2b313a]' : 'border-[#2b313a] bg-[#202630] hover:border-[#474d57]'}`}
            onClick={() => setViewMode('ALL')}
            title="Default (Buy & Sell)"
          >
            <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
              <rect x="1" y="1" width="12" height="5" rx="1" fill="#F6465D" />
              <rect x="1" y="8" width="12" height="5" rx="1" fill="#0ECB81" />
            </svg>
          </button>

          <button 
            className={`w-6 h-5 flex items-center justify-center rounded border transition-colors ${viewMode === 'BIDS' ? 'border-[#fcd535] bg-[#2b313a]' : 'border-[#2b313a] bg-[#202630] hover:border-[#474d57]'}`}
            onClick={() => setViewMode('BIDS')}
            title="Buy Orders Only"
          >
            <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
              <rect x="1" y="1" width="12" height="3" rx="0.5" fill="#0ECB81" />
              <rect x="1" y="5.5" width="12" height="3" rx="0.5" fill="#0ECB81" />
              <rect x="1" y="10" width="12" height="3" rx="0.5" fill="#0ECB81" />
            </svg>
          </button>

          <button 
            className={`w-6 h-5 flex items-center justify-center rounded border transition-colors ${viewMode === 'ASKS' ? 'border-[#fcd535] bg-[#2b313a]' : 'border-[#2b313a] bg-[#202630] hover:border-[#474d57]'}`}
            onClick={() => setViewMode('ASKS')}
            title="Sell Orders Only"
          >
            <svg width="12" height="12" viewBox="0 0 14 14" fill="none">
              <rect x="1" y="1" width="12" height="3" rx="0.5" fill="#F6465D" />
              <rect x="1" y="5.5" width="12" height="3" rx="0.5" fill="#F6465D" />
              <rect x="1" y="10" width="12" height="3" rx="0.5" fill="#F6465D" />
            </svg>
          </button>
        </div>

        {/* Precision Dropdown */}
        <div className="relative flex items-center">
          <select 
            value={precision} 
            onChange={e => setPrecision(Number(e.target.value))}
            className="appearance-none bg-[#202630] border border-[#2b313a] hover:border-[#474d57] text-[#848e9c] hover:text-[#eaecef] text-[11px] font-mono rounded px-2 pr-5 py-0.5 outline-none cursor-pointer"
          >
            <option value={2}>0.01</option>
            <option value={1}>0.1</option>
            <option value={0}>1</option>
          </select>
          <ChevronDown size={11} className="absolute right-1.5 pointer-events-none text-[#848e9c]" />
        </div>
      </div>

      {/* 3. Columns Header */}
      <div className="grid grid-cols-3 px-3.5 py-1 text-[11px] text-[#848e9c] border-b border-[#202630]">
        <span className="text-left">Price ({quoteSymbol})</span>
        <span className="text-right">Amount ({baseSymbol})</span>
        <span className="text-right">Total</span>
      </div>

      {/* 4. Scrollable List */}
      <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
        {/* Asks (Red) */}
        {displayAsks.length > 0 && (
          <div className="flex flex-col">
            {displayAsks.map((ask, idx) => {
              const depthPct = Math.min(100, Math.max(1, (ask.total / maxTotal) * 100));
              return (
                <div 
                  key={`ask-${idx}`} 
                  className="relative grid grid-cols-3 px-3.5 h-[20px] items-center text-[12px] tabular-nums hover:bg-white/[0.03] cursor-pointer"
                >
                  <div 
                    className="absolute right-0 top-0 bottom-0 bg-[#f6465d]/15 pointer-events-none transition-all duration-100" 
                    style={{ width: `${depthPct}%` }} 
                  />
                  <span className="text-left font-medium text-[#F6465D] z-10">{formatPrice(ask.price)}</span>
                  <span className="text-right text-[#eaecef] z-10">{formatAmount(ask.size)}</span>
                  <span className="text-right text-[#848e9c] z-10">{formatTotal(ask.total)}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Middle Live Price Banner */}
        <div className="flex items-center justify-between px-3.5 py-1.5 my-0.5 bg-[#181a20] border-y border-[#202630]">
          <div className="flex items-baseline gap-2">
            <span className={`text-[17px] font-bold tabular-nums ${isUp ? 'text-[#0ECB81]' : 'text-[#F6465D]'}`}>
              {formatPrice(currentPrice || 0)}
            </span>
            <span className={`text-[15px] font-bold ${isUp ? 'text-[#0ECB81]' : 'text-[#F6465D]'}`}>
              {isUp ? '↑' : '↓'}
            </span>
            <span className="text-[12px] text-[#848e9c] tabular-nums">
              ${formatPrice(currentPrice || 0)}
            </span>
          </div>

          <div className="flex items-center text-[#848e9c]">
            <ChevronRight size={14} />
          </div>
        </div>

        {/* Bids (Green) */}
        {displayBids.length > 0 && (
          <div className="flex flex-col">
            {displayBids.map((bid, idx) => {
              const depthPct = Math.min(100, Math.max(1, (bid.total / maxTotal) * 100));
              return (
                <div 
                  key={`bid-${idx}`} 
                  className="relative grid grid-cols-3 px-3.5 h-[20px] items-center text-[12px] tabular-nums hover:bg-white/[0.03] cursor-pointer"
                >
                  <div 
                    className="absolute right-0 top-0 bottom-0 bg-[#0ecb81]/15 pointer-events-none transition-all duration-100" 
                    style={{ width: `${depthPct}%` }} 
                  />
                  <span className="text-left font-medium text-[#0ECB81] z-10">{formatPrice(bid.price)}</span>
                  <span className="text-right text-[#eaecef] z-10">{formatAmount(bid.size)}</span>
                  <span className="text-right text-[#848e9c] z-10">{formatTotal(bid.total)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
