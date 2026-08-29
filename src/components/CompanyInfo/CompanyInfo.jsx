import React from 'react';

export default function CompanyInfo({ details, currentPrice }) {
  if (!details) return null;

  return (
    <div className="bg-[#181a20] border-t border-[#202630] px-4 py-2.5 flex items-center justify-between text-xs text-[#848e9c] select-none shrink-0">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <strong className="text-[#eaecef] font-bold">{details.name}</strong>
          <span className="bg-[#202630] text-[#eaecef] text-[10px] px-1.5 py-0.5 rounded border border-[#2b313a]">
            {details.sector}
          </span>
        </div>

        <div className="hidden md:flex items-center gap-4 text-[11px]">
          <span>Exchange: <strong className="text-[#eaecef]">{details.exchange || 'Binance'}</strong></span>
          <span>Market Cap: <strong className="text-[#eaecef]">{details.marketCap || 'N/A'}</strong></span>
          <span>Currency: <strong className="text-[#eaecef]">{details.currency || 'USD'}</strong></span>
        </div>
      </div>

      <div className="text-[11px] text-[#5e6673] hidden lg:block">
        Real-Time Multi-Feed Trading Terminal
      </div>
    </div>
  );
}
