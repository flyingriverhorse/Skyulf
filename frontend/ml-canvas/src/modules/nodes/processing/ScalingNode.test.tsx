import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ScalingNode } from './ScalingNode';
import type { Recommendation } from '../../../core/api/client';

const source = vi.hoisted(() => ({
  datasetId: 'dataset', loading: false, wide: false, missing: false,
  metrics: null as Record<string, unknown> | null,
  recommendations: [] as Recommendation[],
  nodes: [{ id: 'node', data: {} }, { id: 'dataset', data: { datasetId: 'dataset' } }],
  edges: [{ id: 'edge', source: 'dataset', target: 'node' }],
}));
vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [{}, { datasetId: source.datasetId }] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: (id: string) => ({
  data: source.missing || !id ? null : { columns: {
    age: { name: 'age', dtype: 'Int64' }, score: { name: 'score', dtype: 'FLOAT' },
    number: { name: 'number', dtype: 'number' }, removed: { name: 'removed', dtype: 'int' },
    text: { name: 'text', dtype: 'object' },
  } }, isLoading: source.loading,
}) }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set(['removed']) }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => source.recommendations }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [React.createRef<HTMLDivElement>(), source.wide] }));
vi.mock('../../../core/store/useGraphStore', () => ({ useGraphStore: (selector: (state: unknown) => unknown) => selector({
  nodes: source.nodes, edges: source.edges,
  executionResult: source.metrics ? { node_results: { node: { metrics: source.metrics } } } : null,
}) }));
const Settings = ScalingNode.settings;
type Config = ReturnType<typeof ScalingNode.getDefaultConfig>;
function setup(overrides: Partial<Config> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = React.useState({ ...ScalingNode.getDefaultConfig(), ...overrides });
    return <Settings nodeId="node" config={config} onChange={(next) => { onChange(next); setConfig(next); }} />;
  }
  return { ...render(<Harness />), onChange };
}
beforeEach(() => {
  source.datasetId = 'dataset'; source.loading = false; source.wide = false; source.missing = false;
  source.metrics = null; source.recommendations = [];
  source.nodes = [{ id: 'node', data: {} }, { id: 'dataset', data: { datasetId: 'dataset' } }];
  source.edges = [{ id: 'edge', source: 'dataset', target: 'node' }];
});

describe('ScalingNode public settings', () => {
  it.each([false, true])('filters columns and preserves picker state at wide=%s', (wide) => {
    // Numeric eligibility, dropped columns and local search survive method updates.
    source.wide = wide;
    const { container, onChange } = setup({ columns: ['age'] });
    const picker = screen.getByRole('group', { name: 'Numeric Columns' });
    expect(within(picker).getByRole('checkbox', { name: 'age' })).toBeChecked();
    expect(within(picker).queryByRole('checkbox', { name: 'removed' })).not.toBeInTheDocument();
    expect(within(picker).queryByRole('checkbox', { name: 'text' })).not.toBeInTheDocument();
    fireEvent.click(within(picker).getByRole('checkbox', { name: 'score' }));
    expect(onChange).toHaveBeenLastCalledWith({ ...ScalingNode.getDefaultConfig(), columns: ['age', 'score'] });
    fireEvent.change(within(picker).getByRole('textbox'), { target: { value: 'age' } });
    const method = screen.getByRole('combobox');
    fireEvent.change(method, { target: { value: 'robust' } });
    expect(screen.getByRole('combobox')).toBe(method);
    expect(within(picker).getByRole('textbox')).toHaveValue('age');
    expect(container.firstChild).toHaveClass(wide ? 'overflow-hidden' : 'overflow-y-auto');
  });
  it('shows missing and loading schema states', () => {
    // Missing connections and loading columns remain distinguishable.
    source.datasetId = ''; source.nodes = []; source.missing = true;
    const { rerender } = render(<Settings config={ScalingNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.getByText('Connect a dataset node to see available columns.')).toBeInTheDocument();
    expect(screen.getByText('No numeric columns found')).toBeInTheDocument();
    source.loading = true;
    rerender(<Settings config={ScalingNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.getByText('Loading columns...')).toBeInTheDocument();
  });

  it('preserves defaults, handles, validation and previews', () => {
    // Public metadata and empty-selection messages are serialized contracts.
    expect(ScalingNode.getDefaultConfig()).toEqual({ columns: [], method: 'standard' });
    expect(ScalingNode.inputs).toEqual([{ id: 'in', label: 'Data', type: 'dataset' }]);
    expect(ScalingNode.outputs).toEqual([{ id: 'out', label: 'Scaled Data', type: 'dataset' }]);
    expect(ScalingNode.validate({ columns: [], method: 'standard' })).toEqual({ field: 'columns', isValid: false, message: 'Select at least one column' });
    expect(ScalingNode.validate({ columns: ['age'], method: 'standard' })).toEqual({ field: 'columns', isValid: true, message: undefined });
    expect(ScalingNode.bodyPreview?.({ columns: [], method: 'standard' })).toBe('standard');
    expect(ScalingNode.bodyPreview?.({ columns: ['age'], method: 'robust' })).toBe('robust \u00b7 1 col');
    expect(ScalingNode.bodyPreview?.({ columns: ['age', 'score'], method: 'minmax' })).toBe('minmax \u00b7 2 cols');
  });
  it.each([
    ['standard', 'Center Data (with_mean)', 'with_mean'], ['standard', 'Scale Variance (with_std)', 'with_std'],
    ['robust', 'Center Data (Median)', 'with_centering'], ['robust', 'Scale Data (IQR)', 'with_scaling'],
  ] as const)('updates %s checkbox %s', (method, label, key) => {
    // Nullish checkbox defaults must remain true and clicks serialize false.
    const { onChange } = setup({ method });
    expect(screen.getByRole('checkbox', { name: label })).toBeChecked();
    fireEvent.click(screen.getByRole('checkbox', { name: label }));
    expect(onChange).toHaveBeenLastCalledWith({ columns: [], method, [key]: false });
  });
  it.each([
    ['minmax', 'Feature Range Minimum', 'feature_range_min', 0], ['minmax', 'Feature Range Maximum', 'feature_range_max', 1],
    ['robust', 'Quantile Range Minimum', 'quantile_range_min', 25], ['robust', 'Quantile Range Maximum', 'quantile_range_max', 75],
  ] as const)('preserves numeric parsing for %s %s', (method, label, key, value) => {
    // Empty input currently serializes NaN; extraction must not change that behavior.
    const { onChange } = setup({ method, with_mean: false });
    const input = screen.getByRole('spinbutton', { name: label });
    expect(input).toHaveValue(value);
    fireEvent.change(input, { target: { value: '0' } });
    fireEvent.change(input, { target: { value: '2.5' } });
    expect(onChange).toHaveBeenLastCalledWith({ columns: [], method, with_mean: false, [key]: 2.5 });
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith({ columns: [], method, with_mean: false, [key]: NaN });
  });
  it('retains hidden options through every method', () => {
    // Method switches must only change method and keep saved method-specific fields.
    const initial = { columns: ['age'], with_mean: false, feature_range_max: 0, quantile_range_min: 0 };
    const { onChange } = setup(initial);
    for (const method of ['robust', 'maxabs', 'minmax', 'standard']) {
      fireEvent.change(screen.getByRole('combobox'), { target: { value: method } });
      expect(onChange).toHaveBeenLastCalledWith({ ...ScalingNode.getDefaultConfig(), ...initial, method });
    }
    expect(screen.getByRole('checkbox', { name: 'Center Data (with_mean)' })).not.toBeChecked();
  });
  it.each([
    ['standard', { mean: [0, 1.234], scale: [0] }, '\u03bc=0.00, \u03c3=0.00', '\u03bc=1.23, \u03c3=-', 'Target: \u03bc \u2248 0, \u03c3 \u2248 1'],
    ['minmax', { data_min: [0, 1.234], data_max: [0] }, 'Min=0.00, Max=0.00', 'Min=1.23, Max=-', 'Target: 0 to 1'],
    ['robust', { center: [0, 1.234], scale: [0] }, 'Med=0.00, IQR=0.00', 'Med=1.23, IQR=-', 'Target: Median \u2248 0, IQR \u2248 1'],
    ['maxabs', { max_abs: [0, 1.234] }, 'MaxAbs=0.00', 'MaxAbs=1.23', 'Target: MaxAbs = 1'],
  ] as const)('formats %s feedback including zero and partial arrays', (method, metrics, first, second, target) => {
    // Exact precision and missing secondary values are user-visible execution feedback.
    source.metrics = { columns: ['first', 'second', 'missing'], ...metrics };
    setup({ method });
    expect(screen.getByText(first)).toBeInTheDocument();
    expect(screen.getByText(second)).toBeInTheDocument();
    expect(screen.getByText(target)).toBeInTheDocument();
    expect(screen.getByText('missing').nextSibling).toHaveTextContent('');
  });
  it.each([null, {}, { mean: [1] }, { steps: { one: { details: {} }, two: { details: {} } } }])('omits unavailable metrics %j', (metrics) => {
    // A column list and an unambiguous step are prerequisites for statistics.
    source.metrics = metrics; setup();
    expect(screen.queryByText('Scaling Statistics')).not.toBeInTheDocument();
  });
});
