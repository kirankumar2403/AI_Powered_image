export default function HistoryPanel({ items, error, onSelect }) {
  return (
    <div>
      <h2>History</h2>
      {error && (
        <p className="banner error" role="alert">
          {error}
        </p>
      )}
      {!items?.length && !error && <p className="muted">No saved analyses yet.</p>}
      <ul className="history-list">
        {(items || []).map((item) => (
          <li key={item.analysis_id}>
            <button type="button" className="history-item" onClick={() => onSelect(item.analysis_id)}>
              <span className="history-main">
                <strong>{item.filename}</strong>
                <span className="muted">{new Date(item.created_at).toLocaleString()}</span>
              </span>
              <span className="history-meta">
                <span>{item.quality_label}</span>
                <span>score {item.quality_score}</span>
                <span>{(item.issues || []).map((i) => i.type).join(", ") || "no issues"}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
