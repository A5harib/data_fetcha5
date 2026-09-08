import React, { useEffect, useRef, useState } from 'react';
import { Activity, WifiOff } from 'lucide-react';
import { connectOrderFlowAnalytics } from '../../services/orderFlowBackendService';

// Probability at or above this counts as the model calling a reversal.
const SIGNAL_CUTOFF = 0.5;
// A call is scored against the price this many samples later.
const RESOLVE_AFTER = 20;
// Move needed for the call to count as correct, in percent.
const MOVE_PCT = 0.05;

export default function ReversalPanel({ symbol = 'BTCUSDT' }) {
  const [status, setStatus] = useState('connecting');
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [score, setScore] = useState({ hits: 0, resolved: 0 });

  // Calls waiting on a future price to be judged against.
  const pending = useRef([]);

  useEffect(() => {
    setStatus('connecting');
    setLatest(null);
    setHistory([]);
    setScore({ hits: 0, resolved: 0 });
    pending.current = [];

    const conn = connectOrderFlowAnalytics(
      symbol,
      (data) => {
        setStatus('live');
        const prob = data?.reversal_prediction?.probability ?? 0;
        const price = data?.price ?? 0;
        setLatest({ ...data, probability: prob });

        setHistory((h) => [...h.slice(-59), prob]);

        // Score any call whose horizon has elapsed. The model predicts a
        // reversal, so a large move either way counts; direction is not
        // part of this label.
        if (price > 0) {
          pending.current.forEach((p) => (p.age += 1));
          const due = pending.current.filter((p) => p.age >= RESOLVE_AFTER);
          if (due.length) {
            pending.current = pending.current.filter((p) => p.age < RESOLVE_AFTER);
            const hits = due.filter(
              (p) => Math.abs((price - p.price) / p.price) * 100 >= MOVE_PCT
            ).length;
            setScore((s) => ({ hits: s.hits + hits, resolved: s.resolved + due.length }));
          }
          if (prob >= SIGNAL_CUTOFF) {
            pending.current.push({ price, age: 0 });
          }
        }
      },
      () => setStatus('error')
    );

    return () => conn?.close?.();
  }, [symbol]);

  const prob = latest?.probability ?? 0;
  const pct = (prob * 100).toFixed(1);
  const hot = prob >= SIGNAL_CUTOFF;
  const accuracy = score.resolved ? ((score.hits / score.resolved) * 100).toFixed(0) : null;

  return (
    <div className="border-t border-[#2b313a] bg-[#12161f] p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="flex items-center gap-1.5 text-[#eaecef] font-medium">
          <Activity size={13} className="text-[#fcd535]" />
          ML Reversal Signal
        </span>
        <span
          className={
            status === 'live'
              ? 'text-[#0ecb81]'
              : status === 'error'
                ? 'text-[#f6465d]'
                : 'text-[#848e9c]'
          }
        >
          {status === 'live' ? '● live' : status === 'error' ? <WifiOff size={12} /> : '○ connecting'}
        </span>
      </div>

      {status === 'error' && (
        <div className="text-[#f6465d]">Backend unreachable.</div>
      )}

      {status !== 'error' && (
        <>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-semibold ${hot ? 'text-[#f6465d]' : 'text-[#eaecef]'}`}>
              {pct}%
            </span>
            <span className="text-[#848e9c]">reversal probability</span>
          </div>

          <div className="h-1.5 bg-[#2b313a] rounded mt-2 overflow-hidden">
            <div
              className={`h-full rounded transition-all ${hot ? 'bg-[#f6465d]' : 'bg-[#fcd535]'}`}
              style={{ width: `${Math.min(100, prob * 100)}%` }}
            />
          </div>

          <div className="mt-1.5 text-[#848e9c] truncate">
            {latest?.reversal_prediction?.signal ?? '—'}
          </div>

          {/* Sparkline of recent probabilities, so a flat or pinned model is obvious. */}
          {history.length > 1 && (
            <svg viewBox="0 0 240 40" preserveAspectRatio="none" className="w-full h-10 mt-2">
              <polyline
                fill="none"
                stroke="#fcd535"
                strokeWidth="1.5"
                points={history
                  .map((p, i) => `${(i / (history.length - 1)) * 240},${40 - p * 38}`)
                  .join(' ')}
              />
            </svg>
          )}

          <div className="mt-2 pt-2 border-t border-[#2b313a] flex justify-between text-[#848e9c]">
            <span>
              calls scored: <span className="text-[#eaecef]">{score.resolved}</span>
            </span>
            <span>
              hit rate:{' '}
              <span className={accuracy === null ? 'text-[#848e9c]' : 'text-[#eaecef]'}>
                {accuracy === null ? '—' : `${accuracy}%`}
              </span>
            </span>
          </div>
          <div className="mt-1 text-[10px] text-[#5e6673] leading-tight">
            Scored live: a call above {SIGNAL_CUTOFF} counts as a hit if price moves
            ≥{MOVE_PCT}% within {RESOLVE_AFTER} ticks.
          </div>
        </>
      )}
    </div>
  );
}
