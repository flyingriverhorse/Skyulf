import { useState } from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { ImputationNode } from './ImputationNode';

const sources = vi.hoisted(() => ({
  isWide: false,
  isLoading: false,
  schema: { columns: {
    first: { name: 'amount', dtype: 'float64' },
    second: { name: 'removed', dtype: 'float64' },
    third: { name: 'city', dtype: 'object' },
  } } as { columns: Record<string, { name: string; dtype: string }> } | undefined,
  schemaCall: vi.fn(),
}));

vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: (datasetId: string | undefined) => {
    sources.schemaCall(datasetId);
    return { data: sources.schema, isLoading: sources.isLoading };
  },
}));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => new Set(['removed']),
}));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [null, sources.isWide],
}));

type Config = ReturnType<typeof ImputationNode.getDefaultConfig>;
const Settings = ImputationNode.settings!;

function renderSettings(patch: Partial<Config> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = useState({ ...ImputationNode.getDefaultConfig(), ...patch });
    return <Settings config={config} nodeId="imputer" onChange={next => { onChange(next); setConfig(next); }} />;
  }
  return { ...render(<Harness />), onChange };
}

function selectMethod(method: Config['method']) {
  fireEvent.change(screen.getByRole('combobox', { name: 'Imputation Method' }), { target: { value: method } });
}

beforeEach(() => {
  sources.isWide = false;
  sources.isLoading = false;
  sources.schema = { columns: {
    first: { name: 'amount', dtype: 'float64' }, second: { name: 'removed', dtype: 'float64' },
    third: { name: 'city', dtype: 'object' },
  } };
  sources.schemaCall.mockClear();
  useGraphStore.setState({
    nodes: [
      { id: 'dataset', data: { datasetId: 'dataset-first' }, position: { x: 0, y: 0 } },
      { id: 'middle', data: {}, position: { x: 0, y: 0 } },
      { id: 'cycle', data: {}, position: { x: 0, y: 0 } },
      { id: 'other', data: { datasetId: 'dataset-second' }, position: { x: 0, y: 0 } },
      { id: 'imputer', data: {}, position: { x: 0, y: 0 } },
    ],
    edges: [
      { id: 'a', source: 'dataset', target: 'middle' }, { id: 'b', source: 'middle', target: 'imputer' },
      { id: 'c', source: 'middle', target: 'cycle' }, { id: 'd', source: 'cycle', target: 'middle' },
      { id: 'e', source: 'other', target: 'imputer' },
    ],
    executionResult: null,
  });
});

it('keeps public defaults, validation priority and preview fallback', () => {
  /** Saved graphs and validation focus depend on this exact node contract. */
  const config = ImputationNode.getDefaultConfig();
  expect(config).toEqual({ columns: [], method: 'simple', strategy: 'mean', fill_value: 0,
    n_neighbors: 5, weights: 'uniform', max_iter: 10, estimator: 'bayesian_ridge', random_state: 0 });
  expect(ImputationNode.type).toBe('imputation_node');
  expect(ImputationNode.validate({ ...config, strategy: 'constant', fill_value: '' })).toEqual({
    isValid: false, field: 'columns', message: 'Select at least one column',
  });
  expect(ImputationNode.validate({ ...config, columns: ['amount'], strategy: 'constant', fill_value: '' })).toEqual({
    isValid: false, field: 'fill_value', message: 'Fill value is required for Constant strategy',
  });
  expect(ImputationNode.validate({ ...config, columns: ['amount'], strategy: 'constant' })).toEqual({ isValid: true });
  expect(ImputationNode.bodyPreview!({ ...config, columns: ['amount'], method: 'knn' })).toBe('mean · 1 col');
  expect(ImputationNode.bodyPreview!(config)).toBeNull();
});

it('finds the first upstream dataset through intermediates and filters dropped schema names', () => {
  /** The settings must preserve dataset precedence and display column names rather than schema keys. */
  renderSettings();
  expect(sources.schemaCall).toHaveBeenLastCalledWith('dataset-first');
  const columns = screen.getByRole('group', { name: 'Target Columns' });
  expect(within(columns).getByText('amount')).toBeInTheDocument();
  expect(within(columns).getByText('city')).toBeInTheDocument();
  expect(within(columns).queryByText('removed')).not.toBeInTheDocument();
  expect(within(columns).queryByText('first')).not.toBeInTheDocument();
});

it('keeps method control identity, column search and hidden options across method switches', () => {
  /** Method changes must not clear saved options or the independent column search. */
  const { onChange } = renderSettings({ columns: ['city'], n_neighbors: 7, max_iter: 14 });
  const method = screen.getByRole('combobox', { name: 'Imputation Method' });
  const search = within(screen.getByRole('group', { name: 'Target Columns' })).getByRole('textbox');
  fireEvent.change(search, { target: { value: 'city' } });
  for (const next of ['knn', 'iterative', 'simple'] as const) {
    selectMethod(next);
    expect(screen.getByRole('combobox', { name: 'Imputation Method' })).toBe(method);
    expect(search).toHaveValue('city');
    expect(onChange).toHaveBeenLastCalledWith({ ...ImputationNode.getDefaultConfig(), columns: ['city'], n_neighbors: 7, max_iter: 14, method: next });
  }
  expect(screen.getByRole('combobox', { name: 'Strategy' })).toHaveValue('mean');
});

it.each(['mean', 'median', 'most_frequent', 'constant'] as const)('preserves %s strategy controls and callback', strategy => {
  /** Every simple strategy must remain selectable without modifying unrelated config fields. */
  const { onChange } = renderSettings();
  fireEvent.change(screen.getByRole('combobox', { name: 'Strategy' }), { target: { value: strategy } });
  expect(onChange).toHaveBeenLastCalledWith({ ...ImputationNode.getDefaultConfig(), strategy });
  if (strategy === 'constant') expect(screen.getByRole('textbox', { name: 'Fill Value' })).toHaveValue('');
  else expect(screen.queryByRole('textbox', { name: 'Fill Value' })).not.toBeInTheDocument();
});

it('keeps constant fill input strings and existing numeric-zero display fallback', () => {
  /** The text input forwards raw strings; extraction must not add numeric coercion. */
  const { onChange } = renderSettings({ strategy: 'constant', fill_value: 0 });
  const fill = screen.getByRole('textbox', { name: 'Fill Value' });
  expect(fill).toHaveValue('');
  for (const value of ['0', 'unknown', '']) {
    fireEvent.change(fill, { target: { value } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ fill_value: value }));
  }
});

it.each([
  ['knn', 'Number of Neighbors', 'n_neighbors', 5],
  ['iterative', 'Max Iterations', 'max_iter', 10],
  ['iterative', 'Random State', 'random_state', 0],
] as const)('preserves %s %s display defaults and integer parsing', (method, name, field, display) => {
  /** Empty and decimal inputs must retain the existing safe integer parser behavior. */
  const { onChange } = renderSettings({ method, [field]: 0 });
  const input = screen.getByRole('spinbutton', { name });
  expect(input).toHaveValue(display);
  fireEvent.change(input, { target: { value: '12.9' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: 12 }));
  fireEvent.change(input, { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: 12 }));
});

it('forwards KNN weights and all iterative estimators without clearing other options', () => {
  /** Method-specific selections must preserve configuration for the hidden method. */
  const { onChange } = renderSettings({ method: 'knn' });
  fireEvent.change(screen.getByRole('combobox', { name: 'Weights' }), { target: { value: 'distance' } });
  selectMethod('iterative');
  for (const estimator of ['decision_tree', 'extra_trees', 'knn', 'bayesian_ridge']) {
    fireEvent.change(screen.getByRole('combobox', { name: 'Estimator' }), { target: { value: estimator } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ estimator, weights: 'distance' }));
  }
});

it('shows missing-schema notices only for the corresponding connection state', () => {
  /** A disconnected loading query must not claim it is fetching usable schema. */
  useGraphStore.setState({ nodes: [], edges: [] });
  sources.schema = undefined;
  sources.isLoading = true;
  const { rerender } = render(<Settings config={ImputationNode.getDefaultConfig()} onChange={() => {}} />);
  expect(screen.getByText('Connect a dataset node to see available columns.')).toBeInTheDocument();
  expect(screen.queryByText('Loading schema...')).not.toBeInTheDocument();
  rerender(<Settings config={ImputationNode.getDefaultConfig()} nodeId="imputer" onChange={() => {}} />);
  expect(sources.schemaCall).toHaveBeenLastCalledWith(undefined);
});

it.each([false, true])('renders wrapped feedback exactly once in the wide=%s layout', isWide => {
  /** Feedback and per-column badges must preserve numeric zero, formatting and placement. */
  sources.isWide = isWide;
  useGraphStore.setState({ executionResult: { pipeline_id: 'pipeline', preview_data: [], recommendations: [], status: 'success', node_results: { imputer: { status: 'success', metrics: {
    steps: { impute: { details: { fill_values: { amount: 1.25, city: 'missing' }, missing_counts: { amount: 0, city: 3 }, total_missing: 3 } } },
  } } } } });
  renderSettings();
  expect(screen.getAllByText('Execution Feedback')).toHaveLength(1);
  expect(screen.getByText('1.2500')).toBeInTheDocument();
  expect(screen.getByText('missing')).toBeInTheDocument();
  expect(screen.getByTitle('0 missing values filled')).toHaveTextContent('0');
  expect(screen.getByText('Total Filled:').parentElement).toHaveTextContent('Total Filled: 3');
  const feedback = screen.getByText('Execution Feedback').parentElement!;
  const method = screen.getByRole('combobox', { name: 'Imputation Method' });
  expect(method.parentElement?.parentElement?.contains(feedback)).toBe(isWide);
});

it('shows generic success only when metrics exist without detailed mappings', () => {
  /** Empty metrics and absent metrics have intentionally different feedback visibility. */
  useGraphStore.setState({ executionResult: { pipeline_id: 'pipeline', preview_data: [], recommendations: [], status: 'success', node_results: { imputer: { status: 'success', metrics: {} } } } });
  const { rerender } = render(<Settings config={ImputationNode.getDefaultConfig()} nodeId="imputer" onChange={() => {}} />);
  expect(screen.getByText('Imputation completed successfully.')).toBeInTheDocument();
  rerender(<Settings config={ImputationNode.getDefaultConfig()} nodeId="other" onChange={() => {}} />);
  expect(screen.queryByText('Execution Feedback')).not.toBeInTheDocument();
});

it('filters recommendations and applies only a stable deduplicated column union', () => {
  /** Applying advice adds columns without overwriting the selected imputation strategy. */
  useGraphStore.setState({ executionResult: { pipeline_id: 'pipeline', preview_data: [], status: 'success', node_results: {}, recommendations: [
    { rule_id: 'matching', type: 'imputation', suggested_node_type: 'SimpleImputer', target_columns: ['city', 'amount', 'city'],
      description: 'Fill columns', reasoning: '', suggested_params: { strategy: 'median' }, confidence: 1 },
    { rule_id: 'unrelated', type: 'scaling', suggested_node_type: 'StandardScaler', target_columns: ['amount'],
      description: 'Scale columns', reasoning: '', suggested_params: {}, confidence: 1 },
  ] } });
  const { onChange } = renderSettings({ columns: ['city'] });
  expect(screen.queryByText('Scale columns')).not.toBeInTheDocument();
  fireEvent.click(screen.getByTitle('Apply Recommendation'));
  expect(onChange).toHaveBeenLastCalledWith({ ...ImputationNode.getDefaultConfig(), columns: ['city', 'amount'] });
});
