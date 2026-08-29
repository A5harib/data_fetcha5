import React from 'react';
import { User, Mail, Globe, Phone, Building2, TrendingUp, TrendingDown, Quote, FileText, ExternalLink } from 'lucide-react';

export default function CardView({ data, category }) {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return (
      <div className="empty-state">
        <div className="empty-icon"><FileText size={28} /></div>
        <h3>No matching items found</h3>
        <p>Try adjusting your search query or refreshing the feed.</p>
      </div>
    );
  }

  // Users Card Renderer
  if (category === 'users') {
    return (
      <div className="cards-grid animate-fade-in">
        {data.map((user) => (
          <div key={user.id} className="data-card">
            <div className="card-header">
              <div className="brand-section">
                <div className="card-avatar">
                  {user.name ? user.name.charAt(0) : <User size={20} />}
                </div>
                <div>
                  <div className="card-title">{user.name}</div>
                  <div className="card-subtitle">@{user.username || 'user'}</div>
                </div>
              </div>
              <span className="card-tag">ID: #{user.id}</span>
            </div>

            <div className="card-body">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Mail size={15} style={{ color: 'var(--accent-cyan)' }} />
                <span>{user.email}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Phone size={15} style={{ color: 'var(--accent-emerald)' }} />
                <span>{user.phone}</span>
              </div>
              {user.company && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Building2 size={15} style={{ color: 'var(--accent-amber)' }} />
                  <span>{user.company.name}</span>
                </div>
              )}
            </div>

            <div className="card-footer">
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Globe size={14} /> {user.website || 'N/A'}
              </span>
              <span style={{ color: 'var(--accent-primary)', fontWeight: 500 }}>Active</span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Posts Card Renderer
  if (category === 'posts') {
    return (
      <div className="cards-grid animate-fade-in">
        {data.map((post) => (
          <div key={post.id} className="data-card">
            <div className="card-header">
              <span className="card-tag">Post #{post.id}</span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Author #{post.userId}</span>
            </div>

            <div className="card-title" style={{ textTransform: 'capitalize' }}>
              {post.title}
            </div>

            <div className="card-body">
              <p>{post.body}</p>
            </div>

            <div className="card-footer">
              <span>JSONPlaceholder API</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)' }}>
                Read more <ExternalLink size={13} />
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Crypto Card Renderer
  if (category === 'crypto') {
    const cryptoList = Array.isArray(data) 
      ? data 
      : Object.entries(data).map(([id, info]) => ({
          id,
          name: id.toUpperCase(),
          price: info.usd || info.current_price || 0,
          change: info.usd_24h_change || info.price_change_percentage_24h || 0
        }));

    return (
      <div className="cards-grid animate-fade-in">
        {cryptoList.map((coin) => {
          const isUp = coin.change >= 0;
          return (
            <div key={coin.id} className="data-card">
              <div className="card-header">
                <div className="brand-section">
                  <div className="card-avatar" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-cyan)' }}>
                    {coin.name ? coin.name.substring(0, 3) : 'CRY'}
                  </div>
                  <div>
                    <div className="card-title">{coin.name}</div>
                    <div className="card-subtitle">Live USD Pair</div>
                  </div>
                </div>
                <div 
                  className="card-tag"
                  style={{
                    background: isUp ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                    color: isUp ? '#34d399' : '#fb7185',
                    borderColor: isUp ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'
                  }}
                >
                  {isUp ? <TrendingUp size={13} /> : <TrendingDown size={13} />}
                  {Math.abs(coin.change).toFixed(2)}%
                </div>
              </div>

              <div style={{ margin: '14px 0', fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                ${Number(coin.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
              </div>

              <div className="card-footer">
                <span>CoinGecko Realtime</span>
                <span style={{ color: isUp ? '#34d399' : '#fb7185' }}>
                  {isUp ? 'Bullish 24h' : 'Bearish 24h'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // Quotes Card Renderer
  if (category === 'quotes') {
    return (
      <div className="cards-grid animate-fade-in">
        {data.map((item) => (
          <div key={item.id} className="data-card">
            <div className="card-header">
              <div className="card-avatar" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc' }}>
                <Quote size={18} />
              </div>
              <span className="card-tag">Quote #{item.id}</span>
            </div>

            <div className="card-body" style={{ fontStyle: 'italic', fontSize: '0.95rem', margin: '8px 0' }}>
              "{item.quote || item.text || item.body}"
            </div>

            <div className="card-footer">
              <strong style={{ color: 'var(--text-primary)' }}>— {item.author || 'Unknown'}</strong>
              <span style={{ color: 'var(--accent-primary)' }}>Wisdom</span>
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Fallback for custom or arbitrary structured data
  return (
    <div className="cards-grid animate-fade-in">
      {(Array.isArray(data) ? data : [data]).map((item, idx) => (
        <div key={idx} className="data-card">
          <div className="card-header">
            <span className="card-tag">Result #{idx + 1}</span>
          </div>
          <div className="card-body">
            <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {JSON.stringify(item, null, 2)}
            </pre>
          </div>
        </div>
      ))}
    </div>
  );
}
