import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { FeatureSelectionNode } from './FeatureSelectionNode';

const sources = vi.hoisted(() => ({
  upstream: [] as Record<string, unknown>[],
  loading: false,
  wide: false,
  schema: vi.fn(),
}));

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => sources.upstream }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: (datasetId: string | undefined) => {
    sources.schema(datasetId);
    return {
      data: { columns: { feature: { name: 'feature' }, removed: { name: 'removed' }, target: { name: 'target' } } },
      isLoading: sources.loading,
    };
  },
}));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => new Set(['removed']),
}));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.useRef<HTMLDivElement>(null), sources.wide] as const,
}));

type Config = ReturnType<typeof FeatureSelectionNode.getDefaultConfig>;
const Settings = FeatureSelectionNode.settings;
const node = (id: string, data: Record<string, unknown> = {}) => ({ id, position: { x: 0, y: 0 }, data });
const edge = (source: string, target: string) => ({ id: `${source}-${target}`, source, target });

function renderSettings(overrides: Partial<Config> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = React.useState<Config>({ ...FeatureSelectionNode.getDefaultConfig(), datasetId: 'dataset-one', ...overrides });
    const update = React.useCallback((next: Config) => { onChange(next); setConfig(next); }, []);
    return <Settings config={config} onChange={update} nodeId="selection" />;
  }
  return { ...render(<Harness />), onChange };
}

function changeMethod(method: Config['method']) {
  fireEvent.change(screen.getByRole('combobox', { name: 'Selection Method' }), { target: { value: method } });
}

beforeEach(() => {
  sources.upstream = [];
  sources.loading = false;
  sources.wide = false;
  sources.schema.mockClear();
  useGraphStore.setState({
    nodes: [node('dataset', { datasetId: 'dataset-one' }), node('selection')],
    edges: [edge('dataset', 'selection')],
    executionResult: null,
  });
});

describe('Feature Selection settings', () => {
  it('keeps saved settings and the method control across all ten methods', () => {
    // Switching methods must preserve hidden configuration and the shared control.
    const saved = { target_column: 'target', k: 7, percentile: 25, step: 2, drop_columns: false };
    const { onChange } = renderSettings(saved);
    const methodControl = screen.getByRole('combobox', { name: 'Selection Method' });
    const methods: Config['method'][] = [
      'variance_threshold', 'correlation_threshold', 'select_k_best', 'select_percentile',
      'select_fpr', 'select_fdr', 'select_fwe', 'generic_univariate_select', 'select_from_model', 'rfe',
    ];
    for (const method of methods) {
      changeMethod(method);
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ ...saved, method }));
      expect(screen.getByRole('combobox', { name: 'Selection Method' })).toBe(methodControl);
      expect(screen.getByRole('checkbox', { name: 'Drop Columns' })).not.toBeChecked();
    }
    expect(screen.getByRole('spinbutton', { name: 'K (Number of Features)' })).toHaveValue(7);
    expect(screen.getByRole('spinbutton', { name: 'Step' })).toHaveValue(2);
  });

  it('filters dropped targets and preserves score selection while changing problem types', () => {
    // Schema filtering and task-specific choices must not erase saved scoring options.
    const { onChange } = renderSettings({ score_func: 'chi2' });
    const target = screen.getByRole('combobox', { name: 'Target Column' });
    expect(within(target).queryByRole('option', { name: 'removed' })).not.toBeInTheDocument();
    fireEvent.change(target, { target: { value: 'target' } });
    const score = screen.getByRole('combobox', { name: 'Scoring Function' });
    expect(within(score).getAllByRole('option')).toHaveLength(5);
    fireEvent.change(screen.getByRole('combobox', { name: 'Problem Type' }), { target: { value: 'classification' } });
    expect(within(score).getByRole('option', { name: 'Chi-squared' })).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Problem Type' }), { target: { value: 'regression' } });
    expect(within(score).queryByRole('option', { name: 'Chi-squared' })).not.toBeInTheDocument();
    expect(within(score).getByRole('option', { name: 'Pearson Correlation' })).toBeInTheDocument();
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ target_column: 'target', problem_type: 'regression', score_func: 'chi2' }));
  });

  it('uses the nearest upstream dataset and clears only the dataset when disconnected', () => {
    // Cycles and a stale dataset on this node must not override upstream provenance.
    useGraphStore.setState({
      nodes: [node('far', { datasetId: 'far-data' }), node('middle'), node('near', { datasetId: 'near-data' }), node('selection', { datasetId: 'stale' })],
      edges: [edge('far', 'middle'), edge('middle', 'selection'), edge('near', 'selection'), edge('selection', 'middle')],
    });
    const { onChange } = renderSettings({ datasetId: 'stale', target_column: 'target' });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ datasetId: 'near-data', target_column: 'target' }));
    expect(sources.schema).toHaveBeenLastCalledWith('near-data');
    act(() => useGraphStore.setState({ edges: [] }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ datasetId: undefined, target_column: 'target' }));
    expect(screen.getByText('Connect a dataset node to configure.')).toBeInTheDocument();
    expect(sources.schema).toHaveBeenLastCalledWith(undefined);
  });

  it('shows inherited targets and keeps the last target when inheritance disappears', () => {
    // Upstream target synchronization must preserve its existing disconnect behavior.
    sources.upstream = [{ target_column: 'target' }];
    sources.loading = true;
    const { onChange, rerender } = renderSettings({ target_column: 'feature' });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ target_column: 'target' }));
    expect(screen.getByText('(Auto-detected)')).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Target Column' })).not.toBeInTheDocument();
    expect(screen.getByText('Loading schema...')).toBeInTheDocument();
    sources.upstream = [];
    rerender(<Settings config={{ method: 'select_k_best', datasetId: 'dataset-one', target_column: 'target' }} onChange={onChange} nodeId="selection" />);
    expect(screen.getByRole('combobox', { name: 'Target Column' })).toHaveValue('target');
  });

  it.each([
    ['select_k_best', 'K (Number of Features)', 'k'],
    ['select_percentile', 'Percentile', 'percentile'],
    ['select_from_model', 'Max Features', 'max_features'],
    ['rfe', 'Step', 'step'],
  ] as const)('preserves zero and the last valid integer for %s', (method, label, field) => {
    // Clearing a numeric field during editing retains the existing safe-parser fallback.
    const { onChange } = renderSettings({ method, [field]: 0 });
    const input = screen.getByRole('spinbutton', { name: label });
    expect(input).toHaveValue(0);
    fireEvent.change(input, { target: { value: '12.9' } });
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: 12 }));
  });

  it.each([
    ['variance_threshold', 'Threshold', 'threshold'],
    ['correlation_threshold', 'Threshold', 'threshold'],
    ['select_fdr', 'Alpha (Significance)', 'alpha'],
    ['generic_univariate_select', 'Parameter', 'param'],
  ] as const)('retains float zero and empty parsing for %s', (method, label, field) => {
    // This refactor must not silently change float parsing, including existing NaN on clear.
    const { onChange } = renderSettings({ method, [field]: 0 });
    const input = screen.getByRole('spinbutton', { name: label });
    expect(input).toHaveValue(0);
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: Number.NaN }));
  });

  it('preserves text thresholds, model choice, generic mode and identify-only configuration', () => {
    // Method-specific controls must keep their original serialized field types.
    const { onChange } = renderSettings({ method: 'select_from_model' });
    fireEvent.change(screen.getByRole('textbox', { name: 'Threshold' }), { target: { value: '1.25*mean' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Estimator' }), { target: { value: 'RandomForest' } });
    fireEvent.click(screen.getByRole('checkbox', { name: 'Drop Columns' }));
    changeMethod('generic_univariate_select');
    fireEvent.change(screen.getByRole('combobox', { name: 'Mode' }), { target: { value: 'fwe' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ threshold: '1.25*mean', estimator: 'RandomForest', drop_columns: false, mode: 'fwe' }));
  });

  it.each([false, true])('renders one feedback panel with zero metrics and sorted top five, wide=%s', (wide) => {
    // Layout changes must keep complete feedback and use scores before importances.
    sources.wide = wide;
    useGraphStore.setState({ executionResult: {
      pipeline_id: 'preview', status: 'success', preview_data: null, recommendations: [],
      node_results: { selection: { status: 'success', error: 'Example warning', metrics: {
        dropped_columns: ['zero'], feature_scores: { zero: 0, first: 9, second: 8, third: 7, fourth: 6, fifth: 5 },
        p_values: { zero: 0 }, feature_importances: { zero: 0, ignored: 100 }, variances: { zero: 0 }, ranking: { zero: 0 },
      } } },
    } });
    renderSettings();
    expect(screen.getAllByText('Execution Feedback')).toHaveLength(1);
    for (const text of ['Example warning', 'Score: 0.0000', 'p-val: 0.00e+0', 'Imp: 0.0000', 'Var: 0.0000', 'Rank: 0']) {
      expect(screen.getByText(text)).toBeInTheDocument();
    }
    const topList = screen.getByText('Top 5 Features').nextElementSibling!;
    expect(Array.from(topList.children, row => row.firstElementChild?.textContent)).toEqual(['first', 'second', 'third', 'fourth', 'fifth']);
  });

  it('keeps node validation and body previews for zero-valued saved parameters', () => {
    // Moving the form must preserve the registry contract and zero-valued summaries.
    expect(FeatureSelectionNode.getDefaultConfig()).toEqual({ method: 'select_k_best', k: 10 });
    expect(FeatureSelectionNode.validate({ method: 'variance_threshold' })).toEqual({ isValid: true });
    expect(FeatureSelectionNode.bodyPreview?.({ method: 'rfe', k: 0 })).toBe('rfe · k=0');
    expect(FeatureSelectionNode.bodyPreview?.({ method: 'variance_threshold', threshold: 0 })).toBe('variance_threshold · σ>0');
  });
});

describe('Feature Selection target requirements', () => {
  it.each(['variance_threshold', 'correlation_threshold'] as const)('allows %s without a target', (method) => {
    // Unsupervised selection must remain usable on datasets without a target.
    expect(FeatureSelectionNode.validate({ method })).toEqual({ isValid: true });
    expect(FeatureSelectionNode.validate({ method, target_column: '' })).toEqual({ isValid: true });
  });

  it.each([
    'select_k_best', 'select_percentile', 'select_fpr', 'select_fdr', 'select_fwe',
    'generic_univariate_select', 'select_from_model', 'rfe',
  ] as const)('still requires a target for %s', (method) => {
    // Exempting unsupervised methods must not admit incomplete supervised settings.
    for (const target_column of [undefined, '']) {
      expect(FeatureSelectionNode.validate({ method, target_column })).toEqual({
        isValid: false, field: 'target_column', message: 'Target column is required for this method.',
      });
    }
    expect(FeatureSelectionNode.validate({ method, target_column: 'target' })).toEqual({ isValid: true });
  });
});
