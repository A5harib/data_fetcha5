import React, { useEffect, useState } from 'react';
import { Gauge } from 'lucide-react';

const BASE = (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').replace(/\/$/, '');
const HORIZON = 60;
// The backend recomputes once per closed 1m bar, so polling faster gains nothing.
const POLL_MS = 30_000;

const fmt = (n) => Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export default function ForecastPanel({ symbol = 'BTCUSDT', currentPrice }) {
  const [f, setF] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    // State resets on symbol change via key={symbol} in the parent.
    let alive = true;
    const load = async () => {
      try {
        const res = await fetch(`${BASE}/api/forecast/${symbol}?horizon=${HORIZON}`);
        const body = await res.json();
        if (!alive) return;
        if (!res.ok) {
          setErr(res.status === 404 ? 'No forecaster trained for this symbol.' : body.detail || 'Forecast unavailable.');
          return;
        }
        setF(body);
        setErr(null);
      } catch {
        if (alive) setErr('Backend unreachable.');
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [symbol]);

  const [lo, hi] = f?.volatility?.range80 ?? [0, 0];
  const price = currentPrice || f?.price;
  // Where the live price sits inside the forecast band, 0..1, clamped for the marker.
  const pos = hi > lo && price ? Math.min(1, Math.max(0, (price - lo) / (hi - lo))) : 0.5;
  const dir = f?.direction;
  const oos = f?.model?.oos;

  return (
    <div className="border-t border-[#2b313a] bg-[#12161f] p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="flex items-center gap-1.5 text-[#eaecef] font-medium">
          <Gauge size={13} className="text-[#fcd535]" />
          Next {HORIZON}m forecast
        </span>
        {f && <span className="text-[#5e6673]">{new Date(f.as_of).toLocaleTimeString()}</span>}
      </div>

      {err && <div className="text-[#848e9c]">{err}</div>}
      {!err && !f && <div className="text-[#848e9c]">Loading…</div>}

      {f && (
        <>
          <div className="text-[#848e9c]">Likely range (right 8 in 10 times)</div>
          <div className="flex items-baseline justify-between">
            <span className="text-[#f6465d] font-semibold">{fmt(lo)}</span>
            <span className="text-[#eaecef] text-sm font-semibold">±{f.volatility.range80_pct.toFixed(2)}%</span>
            <span className="text-[#0ecb81] font-semibold">{fmt(hi)}</span>
          </div>
          <div className="relative h-1.5 bg-gradient-to-r from-[#f6465d]/40 via-[#2b313a] to-[#0ecb81]/40 rounded mt-1.5">
            <div className="absolute -top-1 w-0.5 h-3.5 bg-[#fcd535]" style={{ left: `calc(${pos * 100}% - 1px)` }} />
          </div>
          <div className="mt-1.5 flex justify-between text-[#848e9c]">
            <span>
              9 in 10 above <span className="text-[#eaecef]">{fmt(f.volatility.floor90)}</span>
            </span>
            <span>
              9 in 10 below <span className="text-[#eaecef]">{fmt(f.volatility.ceiling90)}</span>
            </span>
          </div>

          <div className="mt-3 flex items-baseline justify-between">
            <span className="text-[#848e9c]">Up or down</span>
            {!dir ? (
              <span className="text-[#848e9c]">not available</span>
            ) : dir.call ? (
              <span className={`font-semibold ${dir.call === 'UP' ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}>
                {dir.call === 'UP' ? '▲ likely up' : '▼ likely down'}
              </span>
            ) : (
              <span className="text-[#848e9c]">no call right now</span>
            )}
          </div>
          {dir && (
            <div className="mt-1 text-[10px] text-[#5e6673] leading-snug">
              Calls only when price is stretched and likely to snap back. Past calls were right{' '}
              {(dir.past_hit_rate * 100).toFixed(0)}% of the time (guessing &quot;up&quot; every time:{' '}
              {(dir.past_hit_rate_always_up * 100).toFixed(0)}%). Misses were bigger than hits (avg{' '}
              {dir.past_avg_loss_pct.toFixed(2)}% vs +{dir.past_avg_win_pct.toFixed(2)}%), so it is a guide, not a
              money machine.
            </div>
          )}

          <div className="mt-2 pt-2 border-t border-[#2b313a] text-[10px] text-[#5e6673] leading-snug">
            Tested on {oos.folds} quarters of data it never saw. Trained through{' '}
            {f.model.trained_through.slice(0, 10)}.
          </div>
        </>
      )}
    </div>
  );
}
