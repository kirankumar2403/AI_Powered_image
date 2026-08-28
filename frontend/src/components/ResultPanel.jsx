function labelClass(label) {
  if (label === "ACCEPTABLE") return "pill ok";
  if (label === "DEGRADED") return "pill warn";
  return "pill bad";
}

export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <div>
        <h2>Result</h2>
        <p className="muted">Run an analysis to see quality, issues, statistics, and explanation.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="result-head">
        <h2>Result</h2>
        <span className={labelClass(result.quality_label)}>{result.quality_label}</span>
      </div>
      <div className="score-row">
        <div>
          <p className="muted">Quality score</p>
          <p className="score">{result.quality_score}</p>
        </div>
        <div>
          <p className="muted">Model confidence</p>
          <p className="score small">{(result.quality_confidence * 100).toFixed(1)}%</p>
        </div>
      </div>

      <h3>Detected issues</h3>
      {result.issues?.length ? (
        <ul className="issue-list">
          {result.issues.map((issue) => (
            <li key={`${issue.type}-${issue.severity}`}>
              <strong>{issue.type}</strong>
              <span className={`sev ${issue.severity}`}>{issue.severity}</span>
              <span className="muted">confidence {(issue.confidence * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No issues above the reporting threshold.</p>
      )}

      <h3>Image statistics</h3>
      <dl className="stats">
        {Object.entries(result.statistics || {}).map(([k, v]) => (
          <div key={k}>
            <dt>{k.replaceAll("_", " ")}</dt>
            <dd>{typeof v === "number" ? v.toFixed(3) : v}</dd>
          </div>
        ))}
      </dl>

      <h3>Explanation</h3>
      <p>{result.explanation?.summary}</p>
      <ul>
        {(result.explanation?.contributing_factors || []).map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      {result.explanation?.feature_importances?.length > 0 && (
        <>
          <h3>Top feature importances (quality RF)</h3>
          <ul className="muted">
            {result.explanation.feature_importances.slice(0, 6).map((f) => (
              <li key={f.feature}>
                {f.feature}: {f.importance}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
