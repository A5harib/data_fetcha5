import React from 'react';
import { Table as TableIcon } from 'lucide-react';

export default function TableView({ data, category }) {
  if (!data || (Array.isArray(data) && data.length === 0)) {
    return (
      <div className="empty-state">
        <div className="empty-icon"><TableIcon size={28} /></div>
        <h3>No table records to display</h3>
      </div>
    );
  }

  // Users Table
  if (category === 'users') {
    return (
      <div className="table-container animate-fade-in">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Username</th>
              <th>Email</th>
              <th>Phone</th>
              <th>Company</th>
              <th>Website</th>
            </tr>
          </thead>
          <tbody>
            {data.map((u) => (
              <tr key={u.id}>
                <td><strong>#{u.id}</strong></td>
                <td>{u.name}</td>
                <td style={{ color: 'var(--accent-cyan)' }}>@{u.username}</td>
                <td>{u.email}</td>
                <td>{u.phone}</td>
                <td>{u.company?.name || '—'}</td>
                <td>{u.website || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // Posts Table
  if (category === 'posts') {
    return (
      <div className="table-container animate-fade-in">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>User ID</th>
              <th>Title</th>
              <th>Body Preview</th>
            </tr>
          </thead>
          <tbody>
            {data.map((p) => (
              <tr key={p.id}>
                <td><strong>#{p.id}</strong></td>
                <td>Author {p.userId}</td>
                <td style={{ fontWeight: 600, maxWidth: '280px' }}>{p.title}</td>
                <td style={{ color: 'var(--text-secondary)', maxWidth: '420px' }}>
                  {p.body?.substring(0, 100)}...
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // Crypto Table
  if (category === 'crypto') {
    const list = Array.isArray(data) 
      ? data 
      : Object.entries(data).map(([id, info]) => ({
          id,
          name: id.toUpperCase(),
          price: info.usd || 0,
          change: info.usd_24h_change || 0
        }));

    return (
      <div className="table-container animate-fade-in">
        <table className="data-table">
          <thead>
            <tr>
              <th>Asset</th>
              <th>Current Price (USD)</th>
              <th>24h % Change</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {list.map((c) => {
              const isUp = c.change >= 0;
              return (
                <tr key={c.id}>
                  <td><strong>{c.name}</strong></td>
                  <td>${Number(c.price).toLocaleString()}</td>
                  <td style={{ color: isUp ? '#34d399' : '#fb7185', fontWeight: 600 }}>
                    {isUp ? '+' : ''}{Number(c.change).toFixed(2)}%
                  </td>
                  <td>{isUp ? '📈 Gaining' : '📉 Dip'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  // Generic flat view
  const items = Array.isArray(data) ? data : [data];
  const keys = items.length > 0 && typeof items[0] === 'object' ? Object.keys(items[0]).slice(0, 6) : [];

  return (
    <div className="table-container animate-fade-in">
      <table className="data-table">
        <thead>
          <tr>
            {keys.map((k) => (
              <th key={k} style={{ textTransform: 'capitalize' }}>{k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((row, i) => (
            <tr key={i}>
              {keys.map((k) => (
                <td key={k}>
                  {typeof row[k] === 'object' ? JSON.stringify(row[k]) : String(row[k] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
