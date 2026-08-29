import React, { useState } from 'react';
import { Send, Zap, Globe2, AlertTriangle } from 'lucide-react';

const PRESETS = [
  { name: 'Random User', url: 'https://randomuser.me/api/?results=5' },
  { name: 'Cat Facts', url: 'https://catfact.ninja/fact' },
  { name: 'Dog API', url: 'https://dog.ceo/api/breeds/image/random' },
  { name: 'Bored API', url: 'https://bored-api.appbrewery.com/random' },
  { name: 'GitHub Zen', url: 'https://api.github.com/zen' }
];

export default function CustomApiExplorer({ onCustomFetch, loading }) {
  const [url, setUrl] = useState('https://catfact.ninja/fact');
  const [method, setMethod] = useState('GET');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (url.trim()) {
      onCustomFetch(url.trim(), method);
    }
  };

  const handleSelectPreset = (presetUrl) => {
    setUrl(presetUrl);
    onCustomFetch(presetUrl, 'GET');
  };

  return (
    <div className="custom-api-box animate-fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Globe2 size={20} color="var(--accent-cyan)" />
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Custom REST API Endpoint Tester</h3>
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
        Query any public REST API URL, inspect status codes, network latency, and response headers in real time.
      </p>

      <form onSubmit={handleSubmit} className="endpoint-bar">
        <select 
          className="method-select" 
          value={method} 
          onChange={(e) => setMethod(e.target.value)}
        >
          <option value="GET">GET</option>
          <option value="POST">POST</option>
        </select>

        <input 
          type="url" 
          className="url-input" 
          placeholder="https://api.example.com/data" 
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          required
        />

        <button 
          type="submit" 
          className="primary-btn" 
          disabled={loading || !url}
        >
          <Send size={16} />
          <span>{loading ? 'Sending...' : 'Send Request'}</span>
        </button>
      </form>

      <div className="presets-container">
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Zap size={14} color="var(--accent-amber)" /> Quick Samples:
        </span>
        {PRESETS.map((p) => (
          <button 
            key={p.name} 
            type="button" 
            className="preset-chip" 
            onClick={() => handleSelectPreset(p.url)}
          >
            {p.name}
          </button>
        ))}
      </div>
    </div>
  );
}
