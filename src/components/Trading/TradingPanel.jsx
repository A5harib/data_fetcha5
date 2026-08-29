import React, { useState } from 'react';
import { DollarSign, Briefcase, History, CheckCircle, TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';

export default function TradingPanel({
  activeSymbol,
  currentPrice,
  portfolioCash,
  positions,
  onExecuteOrder,
  onClosePosition
}) {
  const [orderSide, setOrderSide] = useState('BUY'); // 'BUY' or 'SELL'
  const [orderType, setOrderType] = useState('MARKET'); // 'MARKET' or 'LIMIT'
  const [shares, setShares] = useState(10);
  const [limitPrice, setLimitPrice] = useState(currentPrice || 100);
  const [activeTab, setActiveTab] = useState('TRADE'); // 'TRADE', 'POSITIONS', 'HISTORY'
  const [successMsg, setSuccessMsg] = useState(null);

  const priceToUse = orderType === 'MARKET' ? (currentPrice || 1) : limitPrice;
  const estimatedTotal = (shares * priceToUse).toFixed(2);

  const handleQuickPercent = (pct) => {
    if (!currentPrice || currentPrice <= 0) return;
    const alloc = (portfolioCash * (pct / 100));
    const qty = Math.max(1, Math.floor(alloc / currentPrice));
    setShares(qty);
  };

  const handleSubmitOrder = (e) => {
    e.preventDefault();
    if (shares <= 0) return;

    const success = onExecuteOrder({
      symbol: activeSymbol,
      side: orderSide,
      type: orderType,
      shares: Number(shares),
      price: Number(priceToUse),
      total: Number(estimatedTotal)
    });

    if (success) {
      setSuccessMsg(`Executed ${orderSide} ${shares} shares of ${activeSymbol} @ $${Number(priceToUse).toFixed(2)}`);
      setTimeout(() => setSuccessMsg(null), 3500);
    }
  };

  const activePosition = positions.find(p => p.symbol === activeSymbol);

  return (
    <div className="trading-panel">
      {/* Sub Tabs */}
      <div className="trading-panel-header">
        <div className="tab-group-small">
          <button 
            className={`tab-btn-small ${activeTab === 'TRADE' ? 'active' : ''}`}
            onClick={() => setActiveTab('TRADE')}
          >
            <DollarSign size={13} />
            <span>Place Order</span>
          </button>
          <button 
            className={`tab-btn-small ${activeTab === 'POSITIONS' ? 'active' : ''}`}
            onClick={() => setActiveTab('POSITIONS')}
          >
            <Briefcase size={13} />
            <span>Positions ({positions.length})</span>
          </button>
        </div>

        <div className="portfolio-balance-pill">
          <span>Buying Power:</span>
          <strong>${Number(portfolioCash).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong>
        </div>
      </div>

      {activeTab === 'TRADE' ? (
        <div className="trade-form-container animate-fade-in">
          {/* Buy / Sell Toggle Buttons */}
          <div className="side-toggle">
            <button
              type="button"
              className={`side-btn buy ${orderSide === 'BUY' ? 'active' : ''}`}
              onClick={() => setOrderSide('BUY')}
            >
              Buy {activeSymbol}
            </button>
            <button
              type="button"
              className={`side-btn sell ${orderSide === 'SELL' ? 'active' : ''}`}
              onClick={() => setOrderSide('SELL')}
            >
              Sell {activeSymbol}
            </button>
          </div>

          <form onSubmit={handleSubmitOrder} className="trade-form">
            <div className="form-row">
              <label>Order Type</label>
              <div className="order-type-toggle">
                <button
                  type="button"
                  className={`type-chip ${orderType === 'MARKET' ? 'active' : ''}`}
                  onClick={() => setOrderType('MARKET')}
                >
                  Market
                </button>
                <button
                  type="button"
                  className={`type-chip ${orderType === 'LIMIT' ? 'active' : ''}`}
                  onClick={() => setOrderType('LIMIT')}
                >
                  Limit
                </button>
              </div>
            </div>

            {orderType === 'LIMIT' && (
              <div className="form-row">
                <label>Limit Price ($)</label>
                <input 
                  type="number" 
                  step="0.01" 
                  className="trade-input" 
                  value={limitPrice} 
                  onChange={e => setLimitPrice(Number(e.target.value))}
                  required
                />
              </div>
            )}

            <div className="form-row">
              <label>Quantity (Shares)</label>
              <input 
                type="number" 
                min="1" 
                className="trade-input" 
                value={shares} 
                onChange={e => setShares(Math.max(1, parseInt(e.target.value) || 1))}
                required
              />
            </div>

            {/* Quick Percentage Chips */}
            <div className="quick-pct-chips">
              {[25, 50, 75, 100].map(pct => (
                <button
                  key={pct}
                  type="button"
                  className="pct-chip"
                  onClick={() => handleQuickPercent(pct)}
                >
                  {pct}%
                </button>
              ))}
            </div>

            <div className="order-summary-box">
              <div className="summary-line">
                <span>Est. Price:</span>
                <span>${Number(priceToUse).toFixed(2)}</span>
              </div>
              <div className="summary-line total">
                <span>Total Order Cost:</span>
                <strong>${Number(estimatedTotal).toLocaleString()}</strong>
              </div>
            </div>

            {successMsg && (
              <div className="order-success-banner animate-fade-in">
                <CheckCircle size={14} />
                <span>{successMsg}</span>
              </div>
            )}

            <button 
              type="submit" 
              className={`submit-order-btn ${orderSide === 'BUY' ? 'buy-btn' : 'sell-btn'}`}
              disabled={orderSide === 'BUY' && Number(estimatedTotal) > portfolioCash}
            >
              {orderSide} {shares} {activeSymbol}
            </button>
          </form>
        </div>
      ) : (
        <div className="positions-container animate-fade-in">
          {positions.length === 0 ? (
            <div className="empty-positions">
              <p>No open positions in your paper trading account.</p>
              <button className="primary-btn" onClick={() => setActiveTab('TRADE')} style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
                Open a Position
              </button>
            </div>
          ) : (
            <div className="positions-table-wrap">
              <table className="positions-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Shares</th>
                    <th>Avg Price</th>
                    <th>Current</th>
                    <th>Unrealized P&L</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map(pos => {
                    const curPrice = pos.symbol === activeSymbol ? currentPrice : pos.currentPrice;
                    const pnl = (curPrice - pos.avgPrice) * pos.shares;
                    const pnlPct = ((curPrice - pos.avgPrice) / pos.avgPrice) * 100;
                    const isProfit = pnl >= 0;

                    return (
                      <tr key={pos.symbol}>
                        <td><strong>{pos.symbol}</strong></td>
                        <td>{pos.shares}</td>
                        <td>${pos.avgPrice.toFixed(2)}</td>
                        <td>${curPrice.toFixed(2)}</td>
                        <td style={{ color: isProfit ? '#089981' : '#f23645', fontWeight: 600 }}>
                          {isProfit ? '+' : ''}${pnl.toFixed(2)} ({isProfit ? '+' : ''}{pnlPct.toFixed(2)}%)
                        </td>
                        <td>
                          <button 
                            className="close-pos-btn" 
                            onClick={() => onClosePosition(pos.symbol)}
                            title="Close full position at market"
                          >
                            Close
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
