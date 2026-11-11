import React, { useEffect, useState } from 'react';

type QualityReport = {
  name: string;
  profile: string;
  total_sequences: number;
  total_messages: number;
  labeled_sequences: number;
  label_coverage: number;
  label_counts: { normal: number; anomalous: number; unlabeled: number; invalid: number };
  lengths: { minimum: number; maximum: number; mean: number; median: number; p95: number };
  duplicates: {
    groups: number;
    extra_sequences: number;
    conflicting_groups: number;
    examples: { sequence_ids: string[]; count: number; labels: string[] }[];
  };
  status: string;
  warnings: { code: string; count: number; message: string }[];
};

const number = (value: number) => value.toLocaleString();

export function QualitySummary({ report, runId }: { report: QualityReport; runId: string }) {
  const coverage = Math.round(report.label_coverage * 100);
  return (
    <section className="quality-section" aria-label="Dataset quality report">
      <div className="panel-heading">
        <div>
          <h3>Dataset quality</h3>
          <p>{report.profile} · {report.name}</p>
        </div>
        <a
          className="button compact"
          href={`/api/runs/${encodeURIComponent(runId)}/quality`}
          download={`data-quality-${runId}.json`}
        >
          Export quality report
        </a>
      </div>
      <div className="metrics-grid">
        <div><small>SEQUENCES</small><strong>{number(report.total_sequences)}</strong></div>
        <div><small>MESSAGES</small><strong>{number(report.total_messages)}</strong></div>
        <div><small>REFERENCE LABELS</small><strong>{coverage}%</strong></div>
        <div><small>DUPLICATE COPIES</small><strong>{number(report.duplicates.extra_sequences)}</strong></div>
      </div>
      <div className="quality-coverage">
        <label htmlFor={`coverage-${runId}`}>Reference-label coverage</label>
        <progress id={`coverage-${runId}`} value={coverage} max={100} />
        <span>{number(report.labeled_sequences)} of {number(report.total_sequences)} sequences labeled</span>
      </div>
      <div className="quality-details">
        <div>
          <h4>Label distribution</h4>
          <dl>
            <div><dt>Normal</dt><dd>{number(report.label_counts.normal)}</dd></div>
            <div><dt>Anomalous</dt><dd>{number(report.label_counts.anomalous)}</dd></div>
            <div><dt>Unlabeled</dt><dd>{number(report.label_counts.unlabeled)}</dd></div>
            <div><dt>Invalid labels</dt><dd>{number(report.label_counts.invalid)}</dd></div>
          </dl>
        </div>
        <div>
          <h4>Messages per sequence</h4>
          <dl>
            <div><dt>Minimum</dt><dd>{number(report.lengths.minimum)}</dd></div>
            <div><dt>Median</dt><dd>{number(report.lengths.median)}</dd></div>
            <div><dt>Mean</dt><dd>{report.lengths.mean.toFixed(1)}</dd></div>
            <div><dt>95th percentile</dt><dd>{number(report.lengths.p95)}</dd></div>
            <div><dt>Maximum</dt><dd>{number(report.lengths.maximum)}</dd></div>
          </dl>
        </div>
      </div>
      {report.warnings.length > 0 ? (
        <div className="quality-findings">
          <h4>Review suggested</h4>
          <ul aria-label="Data quality findings">
            {report.warnings.map(warning => (
              <li key={warning.code}>
                <strong>{number(warning.count)}</strong> · {warning.message}
              </li>
            ))}
          </ul>
        </div>
      ) : <p className="note">No issues detected in these checks.</p>}
      {report.duplicates.examples.length > 0 && (
        <details className="quality-duplicates">
          <summary>
            {number(report.duplicates.groups)} duplicate groups ·{' '}
            {number(report.duplicates.conflicting_groups)} with conflicting labels
          </summary>
          <ul>
            {report.duplicates.examples.map((example, index) => (
              <li key={index}>
                <span className="mono">{example.sequence_ids.join(', ')}</span>
                <span> · {example.count} copies · {example.labels.join(' / ') || 'unlabeled'}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <p className="small muted">
        These checks describe the input collection. They do not measure model accuracy or change predictions.
      </p>
    </section>
  );
}

export default function DataQuality({ runId }: { runId: string }) {
  const [report, setReport] = useState<QualityReport | null>(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    // Cancel stale requests when the selected run changes or the panel unmounts.
    const controller = new AbortController();
    setReport(null);
    setError('');
    fetch(`/api/runs/${encodeURIComponent(runId)}/quality`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`Quality report unavailable (${response.status}).`);
        return response.json() as Promise<QualityReport>;
      })
      .then(value => {
        if (!controller.signal.aborted) setReport(value);
      })
      .catch(reason => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Quality report unavailable.');
      });
    return () => controller.abort();
  }, [runId, attempt]);

  if (error) {
    return (
      <div className="inline-error" role="alert">
        {error}
        <button className="text-button" onClick={() => setAttempt(value => value + 1)}>Retry quality report</button>
      </div>
    );
  }
  if (!report) return <p role="status">Loading dataset quality…</p>;
  return <QualitySummary report={report} runId={runId} />;
}
