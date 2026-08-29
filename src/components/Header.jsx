import React from 'react';
import { Database, RefreshCw, Sun, Moon, Sparkles } from 'lucide-react';

export default function Header({ theme, toggleTheme, onRefresh, loading, activeCategory }) {
  return (
    <header className="app-header">
      <div className="header-top">
        <div className="brand-section">
          <div className="logo-badge">
            <Database size={24} />
          </div>
          <div className="brand-info">
            <h1>DataFetch Studio</h1>
            <p>High-performance React API & Data Explorer</p>
          </div>
        </div>

        <div className="header-actions">
          <button 
            className="icon-btn" 
            onClick={toggleTheme} 
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            aria-label="Toggle Theme"
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          <button 
            className="primary-btn" 
            onClick={onRefresh} 
            disabled={loading}
            title="Refetch Current Endpoint"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            <span>{loading ? 'Fetching...' : 'Refetch Data'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
