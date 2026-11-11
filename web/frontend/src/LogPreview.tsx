import {useEffect, useRef, useState} from 'react';
import './log-preview.css';

type PreviewRecord = {
  message: string;
  source: string;
  severity: string;
  timestamp: string | null;
};

type PreviewResult = {
  total_records: number;
  sample_count: number;
  redacted: boolean;
  redaction_counts: Record<string, number>;
  summary: {
    records: number;
    duplicate_messages: number;
    severity_counts: Record<string, number>;
  };
  grouping: {
    sequences: number;
    minimum_length: number | null;
    maximum_length: number | null;
  };
  sample: PreviewRecord[];
};

export default function LogPreview({raw, profile}: {raw: string; profile: string}) {
  const [search, setSearch] = useState('');
  const [minimumSeverity, setMinimumSeverity] = useState('');
  const [redact, setRedact] = useState(true);
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    request.current?.abort();
    setPreview(null);
    setError('');
    setBusy(false);
    return () => request.current?.abort();
  }, [raw, profile, search, minimumSeverity, redact]);

  async function run(format?: 'csv' | 'jsonl') {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError('');
    const body = {
      profile,
      raw_text: raw,
      contains: search,
      minimum_severity: minimumSeverity || null,
      redact,
      sample_limit: 25,
    };
    try {
      const endpoint = format ? `/api/log-preview/export?format=${format}` : '/api/log-preview';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(typeof failure?.detail === 'string' ? failure.detail : 'Unable to preview this log file.');
      }
      if (format) {
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `log-preview.${format}`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      } else {
        const result = await response.json() as PreviewResult;
        if (!controller.signal.aborted) setPreview(result);
      }
    } catch (failure) {
      if (!controller.signal.aborted) setError((failure as Error).message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }

  return <section className="log-preview-panel" aria-label="Log upload preview">
    <div className="log-preview-heading">
      <div><h3>Preview log processing</h3><p>Inspect message groups and a sample before analysis.</p></div>
      <button type="button" className="button compact" disabled={busy} onClick={() => run()}>
        {busy ? 'Processing…' : 'Preview logs'}
      </button>
    </div>
    <div className="log-preview-controls">
      <label>Find text<input value={search} maxLength={200} onChange={event => setSearch(event.target.value)} placeholder="Search message content"/></label>
      <label>Minimum severity<select value={minimumSeverity} onChange={event => setMinimumSeverity(event.target.value)}>
        <option value="">All severities</option>
        {['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'].map(level => <option key={level}>{level}</option>)}
      </select></label>
    </div>
    <label className="log-preview-mask"><input type="checkbox" checked={redact} onChange={event => setRedact(event.target.checked)}/>
      Mask common sensitive fields in the preview and downloads
    </label>
    <p className="log-preview-note">Masking covers common credentials, email addresses, and IPv4 addresses. Review the sample before sharing.</p>
    {error && <div className="inline-error" role="alert">{error}</div>}
    {preview && <div aria-live="polite">
      <dl className="log-preview-stats">
        <div><dt>Input messages</dt><dd>{preview.total_records.toLocaleString()}</dd></div>
        <div><dt>Sequences</dt><dd>{preview.grouping.sequences.toLocaleString()}</dd></div>
        <div><dt>Matching messages</dt><dd>{preview.summary.records.toLocaleString()}</dd></div>
        <div><dt>Repeated messages</dt><dd>{preview.summary.duplicate_messages.toLocaleString()}</dd></div>
      </dl>
      <p className="log-preview-note">Grouping uses all input messages. The sample shows up to 25 matching messages. Timestamps are interpreted as UTC.</p>
      {preview.redacted && <p className="log-preview-note">Masked fields: {Object.values(preview.redaction_counts).reduce((sum, count) => sum + count, 0).toLocaleString()}</p>}
      <div className="log-preview-downloads">
        <button type="button" className="button compact" disabled={busy || !preview.sample_count} onClick={() => run('csv')}>Download sample CSV</button>
        <button type="button" className="button compact" disabled={busy || !preview.sample_count} onClick={() => run('jsonl')}>Download sample JSONL</button>
      </div>
      <div className="log-preview-table">
        <table><thead><tr><th>Timestamp</th><th>Severity</th><th>Message</th></tr></thead>
          <tbody>{preview.sample.map((record, index) => <tr key={index}>
            <td>{record.timestamp || 'Unavailable'}</td><td>{record.severity}</td><td><pre>{record.message}</pre></td>
          </tr>)}</tbody>
        </table>
        {!preview.sample_count && <p>No messages match these filters.</p>}
      </div>
    </div>}
  </section>;
}
