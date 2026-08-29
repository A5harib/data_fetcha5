import React, { useState, useEffect } from 'react';
import { generateOrderBook } from '../../services/polygonService';
import { ArrowUpRight, ArrowDownRight, Layers, Activity } from 'lucide-react';

export default function OrderBook({ currentPrice, activeSymbol, livePriceTick }) {
  const [orderBook, setOrderBook] = useState(() => generateOrderBook(currentPrice || 150));
  const [trades, setTrades] = useState([]);
  const [activeTab, setActiveTab] = useState('BOOK'); // 'BOOK' or 'TRADES'

  // Update orderbook when price changes
  useEffect(() => {
    if (currentPrice) {
      setOrderBook(generateOrderBook(currentPrice));
    }
  }, [currentPrice, activeSymbol]);

  // Handle incoming live ticks to append to trade tape
  useEffect(() => {
    if (!livePriceTick) return;

    const isBuy = livePriceTick.isBuy ?? Math.random() > 0.48;
    const newTrade = {
      id: Date.now() + Math.random(),
      time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      price: livePriceTick.price,
      size: Math.floor(Math.random() * 250) + 10,
      side: isBuy ? 'BUY' : 'SELL'
    };

    setTrades(prev => [newTrade, ...prev.slice(0, 19)]);
  }, [livePriceTick]);

  const maxBidVol = Math.max(...(orderBook.bids.map(b => b.total) || [1]), 1);
  const maxAskVol = Math.max(...(orderBook.asks.map(a => a.total) || [1]), 1);

  return (
    <div className="orderbook-panel">
      <div className="orderbook-header">
        <div className="tab-group-small">
          <button 
            className={`tab-btn-small ${activeTab === 'BOOK' ? 'active' : ''}`}
            onClick={() => setActiveTab('BOOK')}
          >
            <Layers size={13} />
            <span>Order Book</span>
          </button>
          <button 
            className={`tab-btn-small ${activeTab === 'TRADES' ? 'active' : ''}`}
            onClick={() => setActiveTab('TRADES')}
          >
            <Activity size={13} />
            <span>Recent Trades</span>
          </button>
        </div>
      </div>

      {activeTab === 'BOOK' ? (
        <div className="orderbook-content animate-fade-in">
          <div className="book-columns-header">
            <span>Price (USD)</span>
            <span>Size</span>
            <span>Total</span>
          </div>

          {/* Asks (Sells) - Reversed so lowest ask is near spread */}
          <div className="book-asks">
            {orderBook.asks.slice(0, 7).reverse().map((ask, i) => {
              const depthPct = (ask.total / maxAskVol) * 100;
              return (
                <div key={i} className="book-row ask-row">
                  <div className="depth-bar ask-depth" style={{ width: `${depthPct}%` }} />
                  <span className="price ask-price">${ask.price.toFixed(2)}</span>
                  <span className="size">{ask.size}</span>
                  <span className="total">{ask.total}</span>
                </div>
              );
            })}
          </div>

          {/* Spread Indicator */}
          <div className="book-spread">
            <div className="spread-price" style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
              ${Number(currentPrice || 0).toFixed(2)}
            </div>
            <div className="spread-label">Spread: ${orderBook.spread.toFixed(2)}</div>
          </div>

          {/* Bids (Buys) */}
          <div className="book-bids">
            {orderBook.bids.slice(0, 7).map((bid, i) => {
              const depthPct = (bid.total / maxBidVol) * 100;
              return (
                <div key={i} className="book-row bid-row">
                  <div className="depth-bar bid-depth" style={{ width: `${depthPct}%` }} />
                  <span className="price bid-price">${bid.price.toFixed(2)}</span>
                  <span className="size">{bid.size}</span>
                  <span className="total">{bid.total}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="trades-content animate-fade-in">
          <div className="book-columns-header">
            <span>Time</span>
            <span>Price</span>
            <span>Size</span>
          </div>

          <div className="trades-list">
            {trades.map(trade => (
              <div key={trade.id} className="trade-row animate-fade-in">
                <span className="trade-time">{trade.time}</span>
                <span className={`trade-price ${trade.side === 'BUY' ? 'bull-text' : 'bear-text'}`}>
                  ${trade.price.toFixed(2)}
                </span>
                <span className="trade-size">{trade.size}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
