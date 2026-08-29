import React, { useState } from 'react';
import { Copy, Check, Code } from 'lucide-react';

export default function JsonViewer({ rawData }) {
  const [copied, setCopied] = useState(false);

  const formattedJson = JSON.stringify(rawData, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(formattedJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="json-viewer-container animate-fade-in">
      <div className="json-actions">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
          <Code size={16} color="var(--accent-primary)" />
          <span>Raw JSON Response Stream</span>
        </div>

        <button 
          className="tab-btn" 
          onClick={handleCopy} 
          style={{ padding: '6px 14px', fontSize: '0.8rem' }}
          title="Copy payload to clipboard"
        >
          {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
          <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
        </button>
      </div>

      <pre className="json-pre">
        <code>{formattedJson}</code>
      </pre>
    </div>
  );
}
