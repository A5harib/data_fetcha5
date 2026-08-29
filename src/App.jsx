import React, { useState, useEffect } from 'react';
import TerminalHeader from './components/Navigation/TerminalHeader';
import TradingViewChart from './components/Chart/TradingViewChart';
import Watchlist from './components/Watchlist/Watchlist';
import RealMarketFeed from './components/MarketData/RealMarketFeed';
import CompanyInfo from './components/CompanyInfo/CompanyInfo';
import {
  fetchBinanceKlines,
  fetchBinance24hStats,
  connectBinanceMultiStream,
  normalizeBinanceSymbol
} from './services/binanceService';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function App() {
  const [activeSymbol, setActiveSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');

  // Indicators State
  const [indicators, setIndicators] = useState({
    vwap: true,
    rsi: false,
    macd: false,
    sma20: true,
    sma50: false,
    sma200: false,
    ema9: false,
    ema21: false,
    bollinger: false
  });

  const [chartData, setChartData] = useState([]);
  const [volumeData, setVolumeData] = useState([]);
  const [rawBars, setRawBars] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePct, setPriceChangePct] = useState(0);
  const [stats24h, setStats24h] = useState(null);
  const [companyDetails, setCompanyDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Binance Real-Time WebSocket Streaming State
  const [liveKlineUpdate, setLiveKlineUpdate] = useState(null);
  const [orderBook, setOrderBook] = useState(null);
  const [liveTrades, setLiveTrades] = useState([]);

  // Paper Trading Portfolio
  const [portfolioCash, setPortfolioCash] = useState(100000.00);
  const [positions, setPositions] = useState([]);

  const toggleIndicator = (indKey) => {
    setIndicators(prev => ({
      ...prev,
      [indKey]: !prev[indKey]
    }));
  };

  // Update Browser Tab Title dynamically with Current Price & Symbol (e.g. 77,814.01 | BTCUSDT | Binance 24/7)
  useEffect(() => {
    const sym = normalizeBinanceSymbol(activeSymbol);
    if (currentPrice) {
      const formattedPrice = Number(currentPrice).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
      document.title = `${formattedPrice} | ${sym} | Binance 24/7`;
    } else {
      document.title = `${sym} | Binance Terminal`;
    }
  }, [currentPrice, activeSymbol]);

  // Load Historical Bars from Binance Spot API
  const loadMarketData = async () => {
    setLoading(true);
    setError(null);
    setLiveTrades([]);
    setOrderBook(null);

    const normSym = normalizeBinanceSymbol(activeSymbol);

    try {
      const [binanceRes, stats] = await Promise.all([
        fetchBinanceKlines(normSym, timeframe),
        fetchBinance24hStats(normSym)
      ]);

      setChartData(binanceRes.candles);
      setVolumeData(binanceRes.volumes);
      setRawBars(binanceRes.rawBars);
      setCurrentPrice(binanceRes.lastPrice);
      setPriceChange(binanceRes.priceChange);
      setPriceChangePct(binanceRes.priceChangePct);
      setStats24h(stats);

      const isGold = normSym === 'PAXGUSDT';
      setCompanyDetails({
        symbol: normSym,
        name: isGold ? 'Gold (PAX Gold / USDT)' : `${normSym.replace('USDT', '')} / USDT Spot`,
        description: isGold 
          ? 'Paxos Gold (PAXG) is an ERC-20 token backed 1:1 by one fine troy ounce of physical gold stored in Brink’s vaults in London.' 
          : `Real-time Binance 24/7 spot market feed streaming live klines, 100ms Level 2 order book depth, and trade tape.`,
        marketCap: isGold ? '$650M Physical Gold' : normSym.includes('BTC') ? '$1.54T' : '$380B',
        sector: isGold ? 'Precious Metals' : 'Crypto Spot',
        exchange: 'Binance Global Spot',
        currency: 'USDT'
      });
    } catch (err) {
      console.error('Binance Load Error:', err);
      setError(err.message || 'Failed to fetch market data from Binance');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLiveKlineUpdate(null);
    loadMarketData();
  }, [activeSymbol, timeframe]);

  // Connect to 24/7 Live Binance WebSocket
  useEffect(() => {
    const wsConn = connectBinanceMultiStream(activeSymbol, timeframe, {
      onKline: (kline) => {
        setLiveKlineUpdate(kline);
        setCurrentPrice(kline.close);
      },
      onDepth: (depthData) => {
        setOrderBook(depthData);
      },
      onTrade: (trade) => {
        setLiveTrades(prev => [trade, ...prev.slice(0, 29)]);
      },
      onTicker: (tickerStats) => {
        setPriceChange(tickerStats.priceChange);
        setPriceChangePct(tickerStats.priceChangePercent);
      }
    });

    return () => {
      wsConn.disconnect();
    };
  }, [activeSymbol, timeframe]);

  // Paper Trading Orders
  const handleExecuteOrder = (order) => {
    const totalCost = order.price * order.shares;

    if (order.side === 'BUY') {
      if (totalCost > portfolioCash) return false;
      setPortfolioCash(prev => prev - totalCost);

      setPositions(prev => {
        const existing = prev.find(p => p.symbol === order.symbol);
        if (existing) {
          const totalShares = existing.shares + order.shares;
          const totalPaid = (existing.shares * existing.avgPrice) + totalCost;
          const newAvg = totalPaid / totalShares;
          return prev.map(p => p.symbol === order.symbol ? { ...p, shares: totalShares, avgPrice: newAvg } : p);
        } else {
          return [...prev, { symbol: order.symbol, shares: order.shares, avgPrice: order.price }];
        }
      });
      return true;
    } else {
      setPortfolioCash(prev => prev + totalCost);
      setPositions(prev => {
        return prev.map(p => {
          if (p.symbol === order.symbol) {
            const rem = p.shares - order.shares;
            return rem > 0 ? { ...p, shares: rem } : null;
          }
          return p;
        }).filter(Boolean);
      });
      return true;
    }
  };

  const handleClosePosition = (symbol) => {
    const pos = positions.find(p => p.symbol === symbol);
    if (!pos) return;
    const curP = currentPrice || pos.avgPrice;
    const proceeds = curP * pos.shares;
    setPortfolioCash(prev => prev + proceeds);
    setPositions(prev => prev.filter(p => p.symbol !== symbol));
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#12161f] text-[#eaecef] overflow-hidden">
      {/* Top Navbar */}
      <TerminalHeader
        activeSymbol={normalizeBinanceSymbol(activeSymbol)}
        onSelectSymbol={setActiveSymbol}
        timeframe={timeframe}
        onChangeTimeframe={setTimeframe}
        chartType={chartType}
        onChangeChartType={setChartType}
        indicators={indicators}
        onToggleIndicator={toggleIndicator}
        currentPrice={currentPrice}
        priceChange={priceChange}
        priceChangePct={priceChangePct}
        stats24h={stats24h}
      />

      {/* Main Terminal Workspace */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: Markets Watchlist */}
        <Watchlist
          activeSymbol={normalizeBinanceSymbol(activeSymbol)}
          onSelectSymbol={setActiveSymbol}
        />

        {/* Center: TradingView Chart + Bottom Info */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-[#12161f]">
          {loading ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-[#848e9c] text-xs">
              <div className="w-8 h-8 border-2 border-[#2b313a] border-t-[#fcd535] rounded-full animate-spin" />
              <span>Connecting to Binance live 24/7 stream for {normalizeBinanceSymbol(activeSymbol)}...</span>
            </div>
          ) : error ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
              <div className="w-12 h-12 rounded-full bg-[#f6465d]/10 text-[#f6465d] flex items-center justify-center mb-3">
                <AlertCircle size={24} />
              </div>
              <h3 className="text-sm font-bold text-[#eaecef] mb-1">Binance Connection Notice</h3>
              <p className="text-xs text-[#848e9c] max-w-sm mb-4">{error}</p>
              <button 
                onClick={loadMarketData} 
                className="flex items-center gap-1.5 bg-[#fcd535] hover:bg-[#e0be30] text-[#181a20] font-bold text-xs px-4 py-2 rounded transition-colors"
              >
                <RefreshCw size={13} />
                <span>Retry Connection</span>
              </button>
            </div>
          ) : (
            <>
              <TradingViewChart
                symbol={normalizeBinanceSymbol(activeSymbol)}
                timeframe={timeframe}
                chartType={chartType}
                data={chartData}
                volumeData={volumeData}
                indicators={indicators}
                liveKlineUpdate={liveKlineUpdate}
              />

              <CompanyInfo
                details={companyDetails}
                currentPrice={currentPrice}
              />
            </>
          )}
        </div>

        {/* Right: Binance Pro Level 2 Order Book & Market Trades */}
        <RealMarketFeed
          symbol={normalizeBinanceSymbol(activeSymbol)}
          currentPrice={currentPrice}
          priceChange={priceChange}
          priceChangePct={priceChangePct}
          rawBars={rawBars}
          orderBook={orderBook}
          liveTrades={liveTrades}
          isLiveWs={true}
          portfolioCash={portfolioCash}
          positions={positions}
          onExecuteOrder={handleExecuteOrder}
          onClosePosition={handleClosePosition}
        />
      </div>
    </div>
  );
}
