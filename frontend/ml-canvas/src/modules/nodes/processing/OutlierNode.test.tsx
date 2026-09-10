import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { OutlierNode } from './OutlierNode';
import type { Recommendation } from '../../../core/api/client';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

const source = vi.hoisted(() => ({
  datasetId: 'dataset', loading: false, wide: false, missing: false,
  metrics: null as Record<string, unknown> | null,
  recommendations: [] as Recommendation[],
  nodes: [{ id: 'node', data: {} }, { id: 'dataset', data: { datasetId: 'dataset' } }],
  edges: [{ id: 'edge', source: 'dataset', target: 'node' }],
}));
vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [{}, { datasetId: source.datasetId }] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: vi.fn((id: string) => ({
  data: source.missing || !id ? null : { columns: id === 'dataset' ? {
    age: { name: 'age', dtype: 'Int64' }, score: { name: 'score', dtype: 'FLOAT' },
    number: { name: 'number', dtype: 'number' }, removed: { name: 'removed', dtype: 'int' },
    text: { name: 'text', dtype: 'object' },
  } : { upstream: { name: `from-${id}`, dtype: 'Int64' } } }, isLoading: source.loading,
})) }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set(['removed']) }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => source.recommendations }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [React.createRef<HTMLDivElement>(), source.wide] }));
vi.mock('../../../core/store/useGraphStore', () => ({ useGraphStore: (selector: (state: unknown) => unknown) => selector({
  nodes: source.nodes, edges: source.edges,
  executionResult: source.metrics ? { node_results: { node: { metrics: source.metrics } } } : null,
}) }));
const Settings = OutlierNode.settings;
type Config = ReturnType<typeof OutlierNode.getDefaultConfig>;
function setup(overrides: Partial<Config> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = React.useState({ ...OutlierNode.getDefaultConfig(), ...overrides });
    return <Settings nodeId="node" config={config} onChange={(next) => { onChange(next); setConfig(next); }} />;
  }
  return { ...render(<Harness />), onChange };
}
beforeEach(() => {
  vi.mocked(useDatasetSchema).mockClear();
  source.datasetId = 'dataset'; source.loading = false; source.wide = false; source.missing = false;
  source.metrics = null; source.recommendations = [];
  source.nodes = [{ id: 'node', data: {} }, { id: 'dataset', data: { datasetId: 'dataset' } }];
  source.edges = [{ id: 'edge', source: 'dataset', target: 'node' }];
});

describe('OutlierNode public settings', () => {
  it.each([false, true])('filters columns and preserves picker state at wide=%s', (wide) => {
    // Numeric eligibility, dropped columns and local search survive method updates.
    source.wide = wide;
    const { container, onChange } = setup({ columns: ['age'] });
    const picker = screen.getByRole('group', { name: 'Numeric Columns' });
    expect(within(picker).getByRole('checkbox', { name: 'age' })).toBeChecked();
    expect(within(picker).queryByRole('checkbox', { name: 'removed' })).not.toBeInTheDocument();
    expect(within(picker).queryByRole('checkbox', { name: 'text' })).not.toBeInTheDocument();
    fireEvent.click(within(picker).getByRole('checkbox', { name: 'score' }));
    expect(onChange).toHaveBeenLastCalledWith({ ...OutlierNode.getDefaultConfig(), columns: ['age', 'score'] });
    fireEvent.change(within(picker).getByRole('textbox'), { target: { value: 'age' } });
    const method = screen.getByRole('combobox');
    fireEvent.change(method, { target: { value: 'winsorize' } });
    expect(screen.getByRole('combobox')).toBe(method);
    expect(within(picker).getByRole('textbox')).toHaveValue('age');
    expect(container.firstChild).toHaveClass(wide ? 'overflow-hidden' : 'overflow-y-auto');
  });
  it('shows missing and loading schema states', () => {
    // Missing connections and loading columns remain distinguishable.
    source.datasetId = ''; source.nodes = []; source.missing = true;
    const { rerender } = render(<Settings config={OutlierNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.getByText('Connect a dataset node to see available columns.')).toBeInTheDocument();
    expect(screen.getByText('No numeric columns found')).toBeInTheDocument();
    source.loading = true;
    rerender(<Settings config={OutlierNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.getByText('Loading columns...')).toBeInTheDocument();
  });

  it('preserves defaults, handles, validation and previews', () => {
    // All inactive method defaults stay in serialized outlier settings.
    expect(OutlierNode.getDefaultConfig()).toEqual({ method: 'iqr', columns: [], multiplier: 1.5, threshold: 3, lower_percentile: 5, upper_percentile: 95, contamination: 0.01 });
    expect(OutlierNode.inputs).toEqual([{ id: 'in', type: 'dataset', label: 'Dataset' }]);
    expect(OutlierNode.outputs).toEqual([{ id: 'out', type: 'dataset', label: 'Cleaned' }]);
    expect(OutlierNode.validate({ columns: [] })).toEqual({ isValid: false, field: 'columns', message: 'Select at least one column.' });
    expect(OutlierNode.validate({ columns: ['age'] })).toEqual({ isValid: true });
    expect(OutlierNode.bodyPreview?.({ columns: [] })).toBe('IQR');
    expect(OutlierNode.bodyPreview?.({ columns: ['age'], method: 'zscore' })).toBe('ZSCORE \u00b7 1 col');
    expect(OutlierNode.bodyPreview?.({ columns: ['a', 'b'], method: 'winsorize' })).toBe('WINSORIZE \u00b7 2 cols');
  });
  it.each([
    ['iqr', 'Multiplier', 'multiplier', 1.5], ['zscore', 'Threshold (Sigma)', 'threshold', 3],
    ['winsorize', 'Lower Percentile', 'lower_percentile', 5], ['winsorize', 'Upper Percentile', 'upper_percentile', 95],
    ['elliptic_envelope', 'Contamination', 'contamination', 0.01],
  ])('preserves %s numeric defaults and empty parsing', (method, label, key, value) => {
    // Zero is retained by nullish defaults and empty input still emits NaN.
    const { onChange } = setup({ method, [key]: undefined });
    const input = screen.getByRole('spinbutton', { name: String(label) });
    expect(input).toHaveValue(value);
    fireEvent.change(input, { target: { value: '0' } });
    expect(onChange).toHaveBeenLastCalledWith({ ...OutlierNode.getDefaultConfig(), method, [key]: 0 });
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith({ ...OutlierNode.getDefaultConfig(), method, [key]: NaN });
  });
  it('retains hidden parameters across methods', () => {
    // Switching the method must preserve every inactive parameter and picker state.
    const initial = { multiplier: 0, threshold: 7, contamination: 0.2 };
    const { onChange } = setup(initial);
    for (const method of ['zscore', 'winsorize', 'elliptic_envelope', 'iqr']) {
      fireEvent.change(screen.getByRole('combobox'), { target: { value: method } });
      expect(onChange).toHaveBeenLastCalledWith({ ...OutlierNode.getDefaultConfig(), ...initial, method });
    }
    expect(screen.getByRole('spinbutton', { name: 'Multiplier' })).toHaveValue(0);
  });
  it('ignores the current-node dataset while traversing a cycle', () => {
    // A current-node dataset is ignored, and cyclic incoming graphs remain usable.
    source.nodes = [{ id: 'node', data: { datasetId: 'wrong' } }, { id: 'middle', data: {} }, { id: 'dataset', data: { datasetId: 'dataset' } }];
    source.edges = [{ id: 'a', source: 'middle', target: 'node' }, { id: 'b', source: 'node', target: 'middle' }, { id: 'c', source: 'dataset', target: 'middle' }];
    setup();
    expect(useDatasetSchema).toHaveBeenLastCalledWith('dataset');
    expect(screen.getByRole('checkbox', { name: 'age' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'from-wrong' })).not.toBeInTheDocument();
    expect(screen.queryByText('Connect a dataset node to see available columns.')).not.toBeInTheDocument();
  });
  it('prefers a direct upstream dataset over an earlier branch with a deeper dataset', () => {
    // Breadth-first traversal must not return the first dataset found by depth-first traversal.
    source.nodes = [
      { id: 'node', data: { datasetId: 'wrong' } },
      { id: 'middle', data: {} },
      { id: 'deep', data: { datasetId: 'deep' } },
      { id: 'near', data: { datasetId: 'near' } },
    ];
    source.edges = [
      { id: 'a', source: 'middle', target: 'node' },
      { id: 'b', source: 'deep', target: 'middle' },
      { id: 'c', source: 'near', target: 'node' },
    ];
    setup();
    expect(useDatasetSchema).toHaveBeenLastCalledWith('near');
    expect(screen.getByRole('checkbox', { name: 'from-near' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'from-deep' })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'from-wrong' })).not.toBeInTheDocument();
  });
  it('resolves equally near datasets in node order despite reversed edge order', () => {
    // Existing getIncomers ordering chooses the first node, not the first edge.
    source.nodes = [
      { id: 'node', data: {} },
      { id: 'first', data: { datasetId: 'first' } },
      { id: 'second', data: { datasetId: 'second' } },
    ];
    source.edges = [
      { id: 'a', source: 'second', target: 'node' },
      { id: 'b', source: 'first', target: 'node' },
    ];
    setup();
    expect(useDatasetSchema).toHaveBeenLastCalledWith('first');
    expect(screen.getByRole('checkbox', { name: 'from-first' })).toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'from-second' })).not.toBeInTheDocument();
  });
  it.each(['iqr', 'zscore', 'winsorize', 'elliptic_envelope'])('renders %s execution branches precisely', (method) => {
    // The visible metric branches and decimal precision must match original output.
    source.metrics = { rows_removed: 1, rows_remaining: 3, bounds: { value: { lower: -1.234, upper: 2.345 } }, stats: { statistic: { mean: 0, std: 1.234 } }, contamination: 0, warnings: ['Warning text'], values_clipped: 0 };
    setup({ method });
    expect(screen.getByText('Execution Feedback')).toBeInTheDocument();
    expect(screen.getByText('Warning text')).toBeInTheDocument();
    expect(screen.getByText('[-1.23, 2.35]')).toBeInTheDocument();
    expect(screen.getByText('\u03bc=0.00, \u03c3=1.23')).toBeInTheDocument();
    expect(screen.getByText('0.0%')).toBeInTheDocument();
    if (method === 'winsorize') expect(screen.getByText('Values Clipped')).toBeInTheDocument();
    else { expect(screen.getByText('(25.0%)')).toBeInTheDocument(); expect(screen.getByText('3 remaining')).toBeInTheDocument(); }
  });
  it.each([
    ['iqr', 21, 100, 'High data loss (>20%).'], ['iqr', 20, 100, null], ['iqr', 0, 100, 'No outliers detected.'],
    ['winsorize', 0, 100, null], ['elliptic_envelope', 0, 0, 'Stochastic Method'], ['iqr', 1, 0, null],
  ])('derives %s loss recommendations for %s/%s', (method, removed, total, expected) => {
    // Strict >20%, zero-loss suppression and stochastic warnings are separate contracts.
    source.metrics = { rows_removed: removed, rows_total: total }; setup({ method });
    if (expected) expect(screen.getByText(expected)).toBeInTheDocument();
    else expect(screen.queryByText('Recommendations')).not.toBeInTheDocument();
  });
  it.each([['iqr', 'IQR'], ['zscore', 'ZScore'], ['winsorize', 'Winsorize'], ['elliptic_envelope', 'EllipticEnvelope']])('uses legacy %s total for recommendations', (method, prefix) => {
    // Prefix fallback affects recommendations without changing row feedback's own denominator.
    source.metrics = { rows_removed: 21, [`${prefix}_rows_total`]: 100 }; setup({ method });
    expect(screen.getByText('High data loss (>20%).')).toBeInTheDocument();
    expect(screen.queryByText('(21.0%)')).not.toBeInTheDocument();
  });
  it('orders backend recommendations first and keeps them informational', () => {
    // Existing Outlier settings supply no apply callback, so no apply button is mounted.
    source.recommendations = [{ rule_id: 'backend', type: 'info', description: 'Backend first', reasoning: 'Backend reason', suggested_node_type: 'OutlierRemoval', suggested_params: { multiplier: 3 }, confidence: 1, target_columns: ['age'] }];
    source.metrics = { rows_removed: 30, rows_total: 100 };
    const { container, onChange } = setup();
    expect(container.textContent!.indexOf('Backend first')).toBeLessThan(container.textContent!.indexOf('High data loss (>20%).'));
    expect(screen.queryByTitle('Apply Recommendation')).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
  it.each([null, {}, { rows_remaining: 2 }])('omits empty feedback %j', (metrics) => {
    // Remaining-row counts alone do not activate the original feedback panel.
    source.metrics = metrics; setup();
    expect(screen.queryByText('Execution Feedback')).not.toBeInTheDocument();
  });
});
