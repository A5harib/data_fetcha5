import React, { useState } from 'react';
import BinanceOrderBook from './BinanceOrderBook';
import BinanceTradeTape from './BinanceTradeTape';
import { DollarSign, Briefcase, Database, Layers } from 'lucide-react';

export default function RealMarketFeed({
  symbol,
  currentPrice,
  priceChange,
  priceChangePct,
  rawBars = [],
  orderBook = null,
  liveTrades = [],
  isLiveWs = false,
  portfolioCash,
  positions,
  onExecuteOrder,
  onClosePosition
}) {
  const [activeTab, setActiveTab] = useState('LIVE'); // 'LIVE' (Order Book + Trades), 'BARS', 'TRADE', 'POSITIONS'
  const [orderSide, setOrderSide] = useState('BUY');
  const [shares, setShares] = useState(1);
  const [successMsg, setSuccessMsg] = useState(null);

  const estimatedTotal = (shares * (currentPrice || 0)).toFixed(2);

  const handleSubmitOrder = (e) => {
    e.preventDefault();
    if (shares <= 0 || !currentPrice) return;

    const success = onExecuteOrder({
      symbol,
      side: orderSide,
      type: 'MARKET',
      shares: Number(shares),
      price: Number(currentPrice),
      total: Number(estimatedTotal)
    });

    if (success) {
      setSuccessMsg(`Executed ${orderSide} ${shares} ${symbol} @ $${Number(currentPrice).toFixed(2)}`);
      setTimeout(() => setSuccessMsg(null), 3000);
    }
  };

  return (
    <aside className="w-[320px] bg-[#181a20] border-l border-[#202630] flex flex-col h-full overflow-hidden select-none">
      {/* Top Mini Switcher */}
      <div className="flex bg-[#12161f] border-b border-[#202630] p-1 gap-1">
        <button 
          className={`flex-1 flex items-center justify-center gap-1.5 py-1 text-[11px] font-semibold rounded transition-colors ${activeTab === 'LIVE' ? 'bg-[#202630] text-[#eaecef]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
          onClick={() => setActiveTab('LIVE')}
        >
          <Layers size={12} />
          <span>Order Book</span>
        </button>

        <button 
          className={`flex-1 flex items-center justify-center gap-1.5 py-1 text-[11px] font-semibold rounded transition-colors ${activeTab === 'BARS' ? 'bg-[#202630] text-[#eaecef]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
          onClick={() => setActiveTab('BARS')}
        >
          <Database size={12} />
          <span>Raw Bars</span>
        </button>

        <button 
          className={`flex-1 flex items-center justify-center gap-1.5 py-1 text-[11px] font-semibold rounded transition-colors ${activeTab === 'TRADE' ? 'bg-[#202630] text-[#eaecef]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
          onClick={() => setActiveTab('TRADE')}
        >
          <DollarSign size={12} />
          <span>Trade</span>
        </button>

        <button 
          className={`flex-1 flex items-center justify-center gap-1.5 py-1 text-[11px] font-semibold rounded transition-colors ${activeTab === 'POSITIONS' ? 'bg-[#202630] text-[#eaecef]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
          onClick={() => setActiveTab('POSITIONS')}
        >
          <Briefcase size={12} />
          <span>Pos ({positions.length})</span>
        </button>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {activeTab === 'LIVE' ? (
          <div className="flex-1 flex flex-col min-h-0 divide-y divide-[#202630]">
            {/* Top 60%: Binance Pro Order Book */}
            <div className="flex-[3] min-h-0 overflow-hidden">
              <BinanceOrderBook
                symbol={symbol}
                currentPrice={currentPrice}
                priceChange={priceChange}
                orderBook={orderBook}
              />
            </div>

            {/* Bottom 40%: Live Trade Tape */}
            <div className="flex-[2] min-h-0 overflow-hidden">
              <BinanceTradeTape
                symbol={symbol}
                liveTrades={liveTrades}
              />
            </div>
          </div>
        ) : activeTab === 'BARS' ? (
          <div className="flex-1 flex flex-col p-3 overflow-y-auto">
            <div className="text-xs font-semibold text-[#848e9c] mb-2">Historical Aggregates</div>
            <div className="flex flex-col gap-1.5">
              {rawBars.slice(0, 40).map((bar, i) => (
                <div key={i} className="bg-[#202630] border border-[#2b313a] rounded p-2 text-[11px] font-mono">
                  <div className="flex justify-between text-[#848e9c] mb-1">
                    <span>{bar.timestamp}</span>
                    <span className={bar.close >= bar.open ? 'text-[#0ECB81] font-semibold' : 'text-[#F6465D] font-semibold'}>
                      ${Number(bar.close).toFixed(2)}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-1 text-[#eaecef]">
                    <span>O: {Number(bar.open).toFixed(2)}</span>
                    <span>H: {Number(bar.high).toFixed(2)}</span>
                    <span>L: {Number(bar.low).toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : activeTab === 'TRADE' ? (
          <div className="flex-1 p-3.5 flex flex-col gap-3 overflow-y-auto">
            <div className="flex justify-between text-xs bg-[#202630] p-2 rounded border border-[#2b313a]">
              <span className="text-[#848e9c]">Buying Power:</span>
              <strong className="text-[#eaecef] font-mono">${Number(portfolioCash).toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong>
            </div>

            <div className="flex gap-2">
              <button
                type="button"
                className={`flex-1 py-2 rounded text-xs font-bold transition-all ${orderSide === 'BUY' ? 'bg-[#0ECB81] text-white' : 'bg-[#202630] text-[#848e9c] hover:text-[#eaecef]'}`}
                onClick={() => setOrderSide('BUY')}
              >
                Buy {symbol}
              </button>
              <button
                type="button"
                className={`flex-1 py-2 rounded text-xs font-bold transition-all ${orderSide === 'SELL' ? 'bg-[#F6465D] text-white' : 'bg-[#202630] text-[#848e9c] hover:text-[#eaecef]'}`}
                onClick={() => setOrderSide('SELL')}
              >
                Sell {symbol}
              </button>
            </div>

            <form onSubmit={handleSubmitOrder} className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-[#848e9c]">Market Price</label>
                <div className="bg-[#202630] border border-[#2b313a] rounded px-2.5 py-1.5 text-xs font-mono font-bold text-[#eaecef]">
                  ${Number(currentPrice || 0).toFixed(2)}
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[11px] text-[#848e9c]">Quantity</label>
                <input 
                  type="number" 
                  step="any"
                  min="0.001" 
                  className="bg-[#202630] border border-[#2b313a] focus:border-[#fcd535] rounded px-2.5 py-1.5 text-xs font-mono text-[#eaecef] outline-none"
                  value={shares} 
                  onChange={e => setShares(parseFloat(e.target.value) || 0.1)}
                  required
                />
              </div>

              <div className="bg-[#202630] rounded p-2 text-xs flex flex-col gap-1 border border-[#2b313a]">
                <div className="flex justify-between text-[#848e9c]">
                  <span>Total Cost:</span>
                  <strong className="text-[#eaecef] font-mono">${Number(estimatedTotal).toLocaleString()}</strong>
                </div>
              </div>

              {successMsg && (
                <div className="bg-[#0ECB81]/15 border border-[#0ECB81]/30 text-[#0ECB81] text-xs p-2 rounded">
                  {successMsg}
                </div>
              )}

              <button 
                type="submit" 
                className={`py-2 rounded text-xs font-bold text-white transition-opacity ${orderSide === 'BUY' ? 'bg-[#0ECB81]' : 'bg-[#F6465D]'}`}
                disabled={orderSide === 'BUY' && Number(estimatedTotal) > portfolioCash}
              >
                Place {orderSide} Order
              </button>
            </form>
          </div>
        ) : (
          <div className="flex-1 p-3 overflow-y-auto">
            {positions.length === 0 ? (
              <div className="text-center text-[#848e9c] text-xs py-8">
                No open positions.
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {positions.map(pos => {
                  const curPrice = pos.symbol === symbol ? (currentPrice || pos.avgPrice) : pos.avgPrice;
                  const pnl = (curPrice - pos.avgPrice) * pos.shares;
                  const isProfit = pnl >= 0;

                  return (
                    <div key={pos.symbol} className="bg-[#202630] border border-[#2b313a] rounded p-2.5 text-xs font-mono flex flex-col gap-1">
                      <div className="flex justify-between items-center">
                        <strong className="text-[#eaecef]">{pos.symbol}</strong>
                        <span className={`font-bold ${isProfit ? 'text-[#0ECB81]' : 'text-[#F6465D]'}`}>
                          {isProfit ? '+' : ''}${pnl.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between text-[11px] text-[#848e9c]">
                        <span>Qty: {pos.shares}</span>
                        <span>Avg: ${pos.avgPrice.toFixed(2)}</span>
                      </div>
                      <button 
                        className="mt-1 bg-[#F6465D]/15 hover:bg-[#F6465D] text-[#F6465D] hover:text-white border border-[#F6465D]/30 py-1 rounded text-[11px] font-sans font-semibold transition-colors"
                        onClick={() => onClosePosition(pos.symbol)}
                      >
                        Close Position
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
