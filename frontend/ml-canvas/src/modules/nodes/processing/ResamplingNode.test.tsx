import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import type { AnalysisProfile, ColumnProfile, Recommendation } from '../../../core/api/client';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { preprocessingConverters } from '../../../core/utils/pipelineConversion/preprocessing';
import { ResamplingNode } from './ResamplingNode';

const dependencies = vi.hoisted(() => ({
  schema: undefined as AnalysisProfile | undefined,
  recommendations: [] as Recommendation[],
  wide: false,
}));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: vi.fn(() => ({ data: dependencies.schema })) }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => dependencies.recommendations }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, dependencies.wide] }));

type Config = ReturnType<typeof ResamplingNode.getDefaultConfig>;
const Settings = ResamplingNode.settings;
const config = { ...ResamplingNode.getDefaultConfig(), target_column: 'outcome' };

function renderSettings(patch: Partial<Config> = {}) {
  const onChange = vi.fn<(next: Config) => void>();
  const view = render(<Settings config={{ ...config, ...patch }} onChange={onChange} nodeId="resample" />);
  return { ...view, onChange };
}

function setSchema(columns: Partial<ColumnProfile>[]) {
  dependencies.schema = { row_count: 10, column_count: columns.length, columns: Object.fromEntries(columns.map((column, index) => [String(index), {
    name: 'feature', dtype: 'int64', missing_count: 0, missing_ratio: 0, unique_count: 2, ...column,
  }])) };
}

function connectGraph() {
  useGraphStore.setState({
    nodes: [
      { id: 'dataset', position: { x: 0, y: 0 }, data: { datasetId: 'dataset-1' } },
      { id: 'drop', position: { x: 0, y: 0 }, data: { definitionType: 'drop_missing_columns', columns: ['dropped'], config: { target_column: ' upstream ' } } },
      { id: 'resample', position: { x: 0, y: 0 }, data: {} },
    ],
    edges: [{ id: 'data-drop', source: 'dataset', target: 'drop' }, { id: 'drop-resample', source: 'drop', target: 'resample' }],
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  dependencies.schema = undefined;
  dependencies.recommendations = [];
  dependencies.wide = false;
  useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
});

it('renders target suggestions in an application listbox and applies a matching column', async () => {
  /** An anchored DOM menu avoids platform-native popup placement outside the field. */
  setSchema([{ name: 'feature' }, { name: 'outcome' }]);
  const { onChange } = renderSettings({ target_column: 'fea' });
  const input = screen.getByRole('combobox', { name: 'Target Column' });
  fireEvent.focus(input);
  const list = await screen.findByRole('listbox', { name: 'Target column suggestions' });
  expect(input).toHaveAttribute('aria-controls', list.id);
  expect(input).toHaveAttribute('aria-expanded', 'true');
  expect(within(list).queryByRole('option', { name: 'outcome' })).not.toBeInTheDocument();
  fireEvent.click(within(list).getByRole('option', { name: 'feature' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, target_column: 'feature' });
  expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
});

it('keeps target typing and keyboard selection while Escape closes only the suggestions', async () => {
  /** Keyboard users must keep focus and custom target names when the native list is replaced. */
  setSchema([{ name: 'feature' }, { name: 'outcome' }]);
  const onKeyDown = vi.fn();
  function Harness() {
    const [value, setValue] = useState({ ...config, target_column: '' });
    return <Settings config={value} onChange={setValue} nodeId="resample" />;
  }
  render(<Harness />);
  const input = screen.getByRole('combobox', { name: 'Target Column' });
  act(() => { input.focus(); });
  await screen.findByRole('listbox');
  fireEvent.keyDown(input, { key: 'ArrowDown' });
  expect(within(screen.getByRole('listbox')).getByRole('option', { selected: true })).toHaveTextContent('feature');
  fireEvent.keyDown(input, { key: 'Enter' });
  expect(input).toHaveValue('feature');
  expect(input).toHaveFocus();
  fireEvent.click(input);
  await screen.findByRole('listbox');
  document.addEventListener('keydown', onKeyDown);
  try {
    fireEvent.keyDown(input, { key: 'Escape' });
  } finally {
    document.removeEventListener('keydown', onKeyDown);
  }
  expect(input).toHaveFocus();
  expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
  expect(onKeyDown).not.toHaveBeenCalled();
  fireEvent.change(input, { target: { value: 'custom target' } });
  expect(input).toHaveValue('custom target');
});

/** Each method must retain exactly its controls and defaults, including methods without extra parameters. */
it.each([
  { type: 'oversampling', method: 'random_over', fields: {} },
  { type: 'oversampling', method: 'smote', fields: { 'k Neighbors': 5 } },
  { type: 'oversampling', method: 'adasyn', fields: { 'k Neighbors': 5 } },
  { type: 'oversampling', method: 'borderline_smote', fields: { 'k Neighbors': 5, 'm Neighbors': 10, Kind: 'borderline-1' } },
  { type: 'oversampling', method: 'svm_smote', fields: { 'k Neighbors': 5, 'm Neighbors': 10, 'Out Step': 0.5 } },
  { type: 'oversampling', method: 'kmeans_smote', fields: { 'k Neighbors': 5, 'Cluster Balance Threshold': 0.1, 'Density Exponent': 'auto' } },
  { type: 'oversampling', method: 'smote_tomek', fields: {} },
  { type: 'undersampling', method: 'random_under_sampling', fields: { Replacement: false } },
  { type: 'undersampling', method: 'nearmiss', fields: { Version: '1' } },
  { type: 'undersampling', method: 'tomek_links', fields: {} },
  { type: 'undersampling', method: 'edited_nearest_neighbours', fields: { 'n Neighbors': 3, 'Selection Kind': 'all' } },
] satisfies { type: Config['type']; method: string; fields: Record<string, string | number | boolean> }[])('shows the $method controls', ({ type, method, fields }) => {
  const { container } = renderSettings({ type, method });
  const all = ['k Neighbors', 'm Neighbors', 'Kind', 'Out Step', 'Cluster Balance Threshold', 'Density Exponent', 'Replacement', 'Version', 'n Neighbors', 'Selection Kind'];
  for (const name of all) {
    const control = screen.queryByLabelText(name);
    if (Object.prototype.hasOwnProperty.call(fields, name)) {
      const value = (fields as Record<string, string | number | boolean>)[name];
      if (typeof value === 'boolean') expect(control).not.toBeChecked();
      else expect(control).toHaveValue(value);
    } else expect(control).not.toBeInTheDocument();
  }
  expect(container.querySelector('[data-validation-field="target_column"]')).toBeInTheDocument();
});

/** Type changes reset only the method; existing parameters remain available if the user switches back. */
it('preserves type/method selections, full update payloads and backend conversion', () => {
  const { onChange, rerender } = renderSettings({ method: 'svm_smote', out_step: 0 });
  fireEvent.change(screen.getByRole('combobox', { name: 'Resampling Type' }), { target: { value: 'undersampling' } });
  const expected = { ...config, type: 'undersampling' as const, method: 'random_under_sampling', out_step: 0 };
  expect(onChange).toHaveBeenLastCalledWith(expected);
  const converted = preprocessingConverters.get('ResamplingNode')?.({ id: 'resample', position: { x: 0, y: 0 }, data: expected });
  expect(converted).toEqual({ stepType: 'Undersampling', params: expected });
  rerender(<Settings config={expected} onChange={onChange} nodeId="resample" />);
  expect(within(screen.getByRole('combobox', { name: 'Method' })).getAllByRole('option').map(option => option.textContent))
    .toEqual(['Random Under Sampling', 'NearMiss', 'Tomek Links', 'Edited Nearest Neighbours']);
  fireEvent.change(screen.getByRole('combobox', { name: 'Resampling Type' }), { target: { value: 'oversampling' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...expected, type: 'oversampling', method: 'smote' });
  fireEvent.change(screen.getByRole('combobox', { name: 'Method' }), { target: { value: 'nearmiss' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...expected, method: 'nearmiss' });
});

/** Integer input clearing retains the configured value, while explicit zero remains zero. */
it.each([
  { method: 'smote', type: 'oversampling' as const, name: 'k Neighbors', key: 'k_neighbors', value: 7 },
  { method: 'svm_smote', type: 'oversampling' as const, name: 'm Neighbors', key: 'm_neighbors', value: 11 },
  { method: 'edited_nearest_neighbours', type: 'undersampling' as const, name: 'n Neighbors', key: 'n_neighbors', value: 4 },
  { method: 'smote', type: 'oversampling' as const, name: 'Random State', key: 'random_state', value: 42 },
])('preserves integer parsing for $name', ({ method, type, name, key, value }) => {
  const { onChange } = renderSettings({ method, type, [key]: value });
  fireEvent.change(screen.getByRole('spinbutton', { name }), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [key]: value }));
  fireEvent.change(screen.getByRole('spinbutton', { name }), { target: { value: '0' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [key]: 0 }));
});

/** Optional neighbor defaults are display defaults; clearing an absent setting currently emits zero. */
it('retains absent-neighbor clearing semantics', () => {
  const { k_neighbors: omitted, ...withoutNeighbors } = config;
  expect(omitted).toBe(5);
  const onChange = vi.fn();
  render(<Settings config={withoutNeighbors} onChange={onChange} />);
  expect(screen.getByRole('spinbutton', { name: 'k Neighbors' })).toHaveValue(5);
  fireEvent.change(screen.getByRole('spinbutton', { name: 'k Neighbors' }), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...withoutNeighbors, k_neighbors: 0 });
});

/** Select and text edits preserve their distinct string or integer payload types. */
it.each([
  { type: 'oversampling' as const, method: 'borderline_smote', name: 'Kind', key: 'kind', input: 'borderline-2', expected: 'borderline-2' },
  { type: 'undersampling' as const, method: 'nearmiss', name: 'Version', key: 'version', input: '3', expected: 3 },
  { type: 'undersampling' as const, method: 'edited_nearest_neighbours', name: 'Selection Kind', key: 'kind_sel', input: 'mode', expected: 'mode' },
  { type: 'oversampling' as const, method: 'kmeans_smote', name: 'Density Exponent', key: 'density_exponent', input: '2.5', expected: '2.5' },
  { type: 'oversampling' as const, method: 'smote', name: 'Sampling Strategy', key: 'sampling_strategy', input: 'not minority', expected: 'not minority' },
])('emits the $name edit without changing other fields', ({ type, method, name, key, input, expected }) => {
  const { onChange } = renderSettings({ type, method });
  fireEvent.change(screen.getByLabelText(name), { target: { value: input } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, type, method, [key]: expected });
});

/** Replacement uses the checkbox state rather than string or numeric coercion. */
it('emits replacement checkbox changes', () => {
  const { onChange } = renderSettings({ type: 'undersampling', method: 'random_under_sampling' });
  fireEvent.click(screen.getByRole('checkbox', { name: 'Replacement' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, type: 'undersampling', method: 'random_under_sampling', replacement: true });
});

/** Float fields preserve zero and their existing NaN-on-clear callback behavior. */
it.each([
  { method: 'svm_smote', name: 'Out Step', key: 'out_step' },
  { method: 'kmeans_smote', name: 'Cluster Balance Threshold', key: 'cluster_balance_threshold' },
])('retains float parsing for $name', ({ method, name, key }) => {
  const { onChange } = renderSettings({ method, [key]: 0 });
  expect(screen.getByRole('spinbutton', { name })).toHaveValue(0);
  fireEvent.change(screen.getByRole('spinbutton', { name }), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [key]: Number.NaN }));
});

/** Actual upstream traversal must find the ancestor dataset and retain target whitespace and priority. */
it('discovers upstream data and excludes dropped columns from local target suggestions', async () => {
  connectGraph();
  setSchema([{ name: 'class' }, { name: 'dropped' }, { name: 'feature' }]);
  const { onChange } = renderSettings({ target_column: '' });
  expect(useDatasetSchema).toHaveBeenLastCalledWith('dataset-1');
  expect(onChange).toHaveBeenLastCalledWith({ ...config, target_column: ' upstream ' });
  const input = screen.getByRole('combobox', { name: 'Target Column' });
  fireEvent.focus(input);
  const list = await screen.findByRole('listbox', { name: 'Target column suggestions' });
  expect(within(list).getAllByRole('option').map(option => option.textContent)).toEqual(['class', 'feature']);
  expect(screen.getByText('Auto-detected from upstream node.')).toBeInTheDocument();
});

/** Schema target detection honors column order and fills only an empty explicit setting. */
it.each([
  { columns: [{ name: 'Class' }, { name: 'target' }], expected: 'Class' },
  { columns: [{ name: 'feature' }, { name: 'outcome', column_type: 'target' }], expected: 'outcome' },
])('uses the first schema target heuristic ($expected)', ({ columns, expected }) => {
  setSchema(columns);
  const { onChange } = renderSettings({ target_column: '' });
  expect(onChange).toHaveBeenCalledExactlyOnceWith({ ...config, target_column: expected });
});

/** Neither upstream nor schema suggestions overwrite an existing target, even whitespace. */
it.each(['local', '  '])('retains explicit target "%s"', target_column => {
  connectGraph();
  setSchema([{ name: 'target' }]);
  const { onChange } = renderSettings({ target_column });
  expect(onChange).not.toHaveBeenCalled();
});

/** Recommendation application merges its payload without discarding form state or resetting its visibility. */
it('applies recommendations and keeps the section collapsed after a method change', () => {
  dependencies.recommendations = [{ rule_id: 'balance', type: 'resampling', description: 'Balance classes', target_columns: [], suggested_node_type: 'ResamplingNode', suggested_params: { method: 'adasyn', k_neighbors: 3 }, confidence: 1, reasoning: 'Imbalanced classes' }];
  const { onChange, rerender } = renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Apply Recommendation' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, method: 'adasyn', k_neighbors: 3 });
  fireEvent.click(screen.getByRole('button', { name: 'Recommendations (1)' }));
  rerender(<Settings config={{ ...config, method: 'svm_smote' }} onChange={onChange} nodeId="resample" />);
  expect(screen.queryByRole('button', { name: 'Apply Recommendation' })).not.toBeInTheDocument();
});

/** Last-run metrics recursively retain zero, false, null, nested keys and error text. */
it('renders nested execution metrics and empty-result states', () => {
  useGraphStore.setState({ executionResult: { pipeline_id: 'pipeline', status: 'success', preview_data: null, recommendations: [], node_results: {
    resample: { metrics: { before_count: 0, balanced: false, summary: { minority_count: 4, missing: null } }, error: 'partial failure' },
  } } });
  renderSettings();
  expect(screen.getByText('Last Run Results')).toBeInTheDocument();
  expect(screen.getByText('before count:').nextSibling).toHaveTextContent('0');
  expect(screen.getByText('balanced:').nextSibling).toHaveTextContent('false');
  expect(screen.getByText('missing:').nextSibling).toHaveTextContent('null');
  expect(screen.getByText('Error: partial failure')).toBeInTheDocument();
  act(() => useGraphStore.setState({ executionResult: { pipeline_id: 'pipeline', status: 'success', preview_data: null, recommendations: [], node_results: { resample: {} } } }));
  expect(screen.getByText('No metrics available')).toBeInTheDocument();
});

/** Stable field identity preserves edits and focus while parameters and layout update. */
it('keeps field focus through settings updates and responsive layout changes', () => {
  function Harness() {
    const [current, setCurrent] = useState(config);
    return <Settings config={current} onChange={setCurrent} nodeId="resample" />;
  }
  const { rerender } = render(<Harness />);
  const target = screen.getByRole('combobox', { name: 'Target Column' });
  target.focus();
  fireEvent.change(target, { target: { value: 'updated' } });
  dependencies.wide = true;
  rerender(<Harness />);
  expect(screen.getByRole('combobox', { name: 'Target Column' })).toBe(target);
  expect(target).toHaveFocus();
  expect(target).toHaveValue('updated');
});

/** Public defaults, preview and validation retain saved-canvas compatibility and existing bounds. */
it('retains the public node definition and method-specific validation', () => {
  expect(ResamplingNode.getDefaultConfig()).toEqual({ type: 'oversampling', method: 'smote', target_column: '', sampling_strategy: 'auto', random_state: 42, k_neighbors: 5, replacement: false, version: 1, n_neighbors: 3, kind_sel: 'all' });
  expect(ResamplingNode.bodyPreview?.(config)).toBe('SMOTE → outcome');
  expect(ResamplingNode.validate({ ...config, target_column: '' })).toEqual({ isValid: false, field: 'target_column', message: 'Target column is required for resampling.' });
  expect(ResamplingNode.validate({ ...config, k_neighbors: 0 })).toEqual({ isValid: false, field: 'k_neighbors', message: 'k_neighbors must be at least 1.' });
  expect(ResamplingNode.validate({ ...config, method: 'smote_tomek', k_neighbors: 0 })).toEqual({ isValid: true });
  expect(ResamplingNode.validate({ ...config, type: 'undersampling', k_neighbors: 0 })).toEqual({ isValid: true });
});
