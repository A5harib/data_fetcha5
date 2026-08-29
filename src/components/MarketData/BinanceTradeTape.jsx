import React from 'react';

export default function BinanceTradeTape({ liveTrades = [], symbol = 'BTCUSDT' }) {
  const baseSymbol = symbol.replace('USDT', '').replace('USD', '').replace('X:', '');
  const quoteSymbol = 'USDT';

  return (
    <div className="flex flex-col bg-[#181a20] text-[#eaecef] select-none h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3.5 pt-2.5 pb-1.5">
        <span className="text-[13px] font-semibold text-[#eaecef]">Market Trades</span>
        <span className="text-[11px] text-[#0ecb81] font-medium flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#0ecb81] animate-pulse" />
          Live WS
        </span>
      </div>

      {/* Column Headers */}
      <div className="grid grid-cols-3 px-3.5 py-1 text-[11px] text-[#848e9c] border-b border-[#202630]">
        <span className="text-left">Price ({quoteSymbol})</span>
        <span className="text-right">Amount ({baseSymbol})</span>
        <span className="text-right">Time</span>
      </div>

      {/* Live Trades Stream */}
      <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
        {liveTrades.length === 0 ? (
          <div className="p-4 text-center text-[#848e9c] text-xs">
            Connecting to live trade stream...
          </div>
        ) : (
          liveTrades.map((trade, i) => (
            <div 
              key={trade.id || i}
              className="grid grid-cols-3 px-3.5 h-[20px] items-center text-[12px] tabular-nums hover:bg-white/[0.03]"
            >
              <span className={`text-left font-medium ${trade.side === 'BUY' ? 'text-[#0ECB81]' : 'text-[#F6465D]'}`}>
                {Number(trade.price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span className="text-right text-[#eaecef]">
                {Number(trade.size).toFixed(5)}
              </span>
              <span className="text-right text-[#848e9c] text-[11px]">
                {trade.time}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
