import React from 'react';
import { Activity, Clock, Layers, CheckCircle2, AlertCircle } from 'lucide-react';

export default function MetricsBar({ status, latency, itemCount, payloadSize }) {
  const isOk = status >= 200 && status < 300;

  return (
    <div className="metrics-bar">
      <div className={`metric-pill ${isOk ? 'status-badge-ok' : 'status-badge-error'}`}>
        {isOk ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
        <span>Status: <strong>{status ? `${status} ${isOk ? 'OK' : 'Error'}` : 'Idle'}</strong></span>
      </div>

      <div className="metric-pill">
        <Clock size={14} />
        <span>Latency: <strong>{latency !== null ? `${latency}ms` : '--'}</strong></span>
      </div>

      <div className="metric-pill">
        <Layers size={14} />
        <span>Records: <strong>{itemCount}</strong></span>
      </div>

      <div className="metric-pill">
        <Activity size={14} />
        <span>Size: <strong>{payloadSize || '--'}</strong></span>
      </div>
    </div>
  );
}
