import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { BINANCE_MARKETS } from '../../services/binanceService';

export default function Watchlist({ activeSymbol, onSelectSymbol }) {
  const [filterQuery, setFilterQuery] = useState('');
  const [activeTab, setActiveTab] = useState('ALL');

  const filtered = BINANCE_MARKETS.filter(item => {
    const matchesSearch = item.symbol.toLowerCase().includes(filterQuery.toLowerCase()) ||
                          item.name.toLowerCase().includes(filterQuery.toLowerCase());
    if (!matchesSearch) return false;

    if (activeTab === 'METALS') return item.symbol.includes('PAXG');
    if (activeTab === 'MAJOR') return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'].includes(item.symbol);
    return true;
  });

  return (
    <aside className="w-[220px] bg-[#181a20] border-r border-[#202630] flex flex-col h-full overflow-hidden select-none shrink-0">
      {/* Search Header */}
      <div className="p-2.5 border-b border-[#202630] flex flex-col gap-2">
        <div className="relative">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#848e9c]" />
          <input 
            type="text" 
            placeholder="Search markets..." 
            value={filterQuery}
            onChange={e => setFilterQuery(e.target.value)}
            className="w-full bg-[#202630] border border-[#2b313a] focus:border-[#fcd535] rounded px-2 pl-7 py-1 text-xs text-[#eaecef] placeholder-[#5e6673] outline-none"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex bg-[#12161f] p-0.5 rounded gap-1 border border-[#202630]">
          {['ALL', 'MAJOR', 'METALS'].map(tab => (
            <button
              key={tab}
              className={`flex-1 py-1 text-[10px] font-bold rounded transition-colors ${activeTab === tab ? 'bg-[#202630] text-[#eaecef]' : 'text-[#848e9c] hover:text-[#eaecef]'}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto divide-y divide-[#202630]/60">
        {filtered.map(item => {
          const isSelected = activeSymbol.toUpperCase() === item.symbol.toUpperCase();

          return (
            <div
              key={item.symbol}
              className={`px-3 py-2.5 flex items-center justify-between cursor-pointer transition-colors hover:bg-[#202630] ${isSelected ? 'bg-[#202630] border-l-2 border-[#fcd535]' : ''}`}
              onClick={() => onSelectSymbol(item.symbol)}
            >
              <div className="flex flex-col">
                <div className="text-xs font-bold text-[#eaecef] flex items-center gap-1">
                  {item.symbol}
                  <span className="text-[9px] font-normal text-[#0ecb81] bg-[#0ecb81]/10 px-1 rounded">
                    24/7
                  </span>
                </div>
                <div className="text-[11px] text-[#848e9c] truncate max-w-[110px]">{item.name}</div>
              </div>

              <span className="text-[10px] text-[#848e9c] bg-[#12161f] px-1.5 py-0.5 rounded border border-[#2b313a]">
                {item.sector}
              </span>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
