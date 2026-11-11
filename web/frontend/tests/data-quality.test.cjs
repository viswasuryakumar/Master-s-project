const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const { test } = require('node:test');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const ts = require('typescript');

const filename = path.resolve(__dirname, '../src/DataQuality.tsx');
const output = ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
});
const loaded = new Module(filename, module);
loaded.filename = filename;
loaded.paths = Module._nodeModulePaths(path.dirname(filename));
loaded._compile(output.outputText, filename);
const { QualitySummary, default: DataQuality } = loaded.exports;

function report(overrides = {}) {
  return {
    name: 'Upload sample', profile: 'HDFS', total_sequences: 2, total_messages: 6,
    labeled_sequences: 1, label_coverage: 0.5,
    label_counts: { normal: 1, anomalous: 0, unlabeled: 1, invalid: 0 },
    lengths: { minimum: 2, maximum: 4, mean: 3, median: 3, p95: 4 },
    duplicates: { groups: 0, extra_sequences: 0, conflicting_groups: 0, examples: [] },
    status: 'ready', warnings: [], ...overrides,
  };
}

const render = value => renderToStaticMarkup(React.createElement(QualitySummary, { report: value, runId: 'run-1' }));

test('shows coverage, counts and sequence statistics', () => {
  const html = render(report());
  assert.match(html, /50%/);
  assert.match(html, /1 of 2 sequences labeled/);
  assert.match(html, /95th percentile/);
  assert.match(html, /No issues detected/);
  assert.match(html, /\/api\/runs\/run-1\/quality/);
});

test('empty collections show zero coverage without NaN or infinity', () => {
  const html = render(report({ total_sequences: 0, total_messages: 0, label_coverage: 0, labeled_sequences: 0 }));
  assert.match(html, /0%/);
  assert.doesNotMatch(html, /NaN|Infinity/);
});

test('findings and conflicting duplicate labels are visible', () => {
  const html = render(report({
    warnings: [{ code: 'conflicting_labels', count: 1, message: 'Conflicting labels need review.' }],
    duplicates: { groups: 1, extra_sequences: 1, conflicting_groups: 1, examples: [{ sequence_ids: ['a', 'b'], count: 2, labels: ['normal', 'anomalous'] }] },
  }));
  assert.match(html, /Review suggested/);
  assert.match(html, /Conflicting labels need review/);
  assert.match(html, /a, b/);
  assert.match(html, /normal \/ anomalous/);
});

test('sequence identifiers render as text', () => {
  const html = render(report({
    duplicates: { groups: 1, extra_sequences: 1, conflicting_groups: 0, examples: [{ sequence_ids: ['<img src=x>'], count: 2, labels: [] }] },
  }));
  assert.match(html, /&lt;img src=x&gt;/);
  assert.doesNotMatch(html, /<img src=x>/);
});

test('the panel has an initial loading state', () => {
  const html = renderToStaticMarkup(React.createElement(DataQuality, { runId: 'run-1' }));
  assert.match(html, /role="status"/);
  assert.match(html, /Loading dataset quality/);
});
