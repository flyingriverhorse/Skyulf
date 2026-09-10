import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { EncodingNode } from './EncodingNode';

const sources = vi.hoisted(() => ({
  datasetId: 'dataset-1' as string | undefined,
  isLoading: false,
  columns: {
    city: { name: 'city', dtype: 'object' },
    removed: { name: 'removed', dtype: 'object' },
    target: { name: 'target', dtype: 'int64' },
  },
}));

vi.mock('../../../core/hooks/useUpstreamData', () => ({
  useUpstreamData: () => [{ datasetId: sources.datasetId }],
}));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: () => ({ data: { columns: sources.columns }, isLoading: sources.isLoading }),
}));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => new Set(['removed']),
}));

type Config = ReturnType<typeof EncodingNode.getDefaultConfig>;
const Settings = EncodingNode.settings!;

function renderSettings(overrides: Partial<Config> = {}) {
  const onChange = vi.fn();
  function Harness() {
    const [config, setConfig] = React.useState({ ...EncodingNode.getDefaultConfig(), ...overrides });
    return <Settings config={config} nodeId="encoder" onChange={(next) => { onChange(next); setConfig(next); }} />;
  }
  return { ...render(<Harness />), onChange };
}

function selectMethod(method: Config['method']) {
  fireEvent.change(screen.getByRole('combobox', { name: 'Encoding Method' }), { target: { value: method } });
}

beforeEach(() => {
  sources.datasetId = 'dataset-1';
  sources.isLoading = false;
  useGraphStore.setState({ executionResult: null });
});

describe('EncodingNode settings behavior', () => {
  it.each(['__proto__', 'constructor', 'toString', 'unknown-method'])(
    'leaves method-specific content empty for imported unsupported method %s', (method) => {
      // Imported configs may contain arbitrary strings, including inherited object keys.
      const config = EncodingNode.getDefaultConfig();
      Reflect.set(config, 'method', method);
      const { container } = render(<Settings config={config} onChange={() => {}} />);
      const control = screen.getByRole('combobox', { name: 'Encoding Method' });
      expect(control.parentElement?.querySelector('p')).toBeEmptyDOMElement();
      expect(container.querySelector('.grid.gap-4')?.lastElementChild).toBeEmptyDOMElement();
      expect(screen.getByRole('group', { name: 'Columns to Encode' })).toBeInTheDocument();
    },
  );

  it('preserves saved options, column search and method control identity across all seven methods', () => {
    // Switching methods must not erase hidden settings or remount the shared picker.
    const initial = { columns: ['city'], max_categories: 37, n_features: 16, missing_code: 0 };
    const { onChange } = renderSettings(initial);
    const methodControl = screen.getByRole('combobox', { name: 'Encoding Method' });
    const picker = screen.getByRole('group', { name: 'Columns to Encode' });
    fireEvent.change(within(picker).getByRole('textbox'), { target: { value: 'city' } });
    const methods: Config['method'][] = ['dummy', 'label', 'ordinal', 'target', 'woe', 'hash', 'onehot'];
    for (const method of methods) {
      selectMethod(method);
      expect(onChange).toHaveBeenLastCalledWith({ ...EncodingNode.getDefaultConfig(), ...initial, method });
      expect(screen.getByRole('combobox', { name: 'Encoding Method' })).toBe(methodControl);
      expect(within(picker).getByRole('textbox')).toHaveValue('city');
    }
    expect(screen.getByRole('spinbutton', { name: 'Max Categories' })).toHaveValue(37);
  });

  it('updates one-hot options independently and retains drop-first when switching to dummy', () => {
    // Checkbox defaults and shared settings must keep the exact serialized config.
    const { onChange } = renderSettings();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Drop Original Column' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Drop First Category' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include missing as category' }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Handle Unknown' }), { target: { value: 'error' } });
    selectMethod('dummy');
    expect(screen.queryByRole('checkbox', { name: 'Drop Original Column' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Drop First Category' })).toBeChecked();
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
      method: 'dummy', drop_original: false, drop_first: true, include_missing: true, handle_unknown: 'error',
    }));
  });

  it.each([
    ['onehot', 'Max Categories', 'max_categories'],
    ['hash', 'Number of Features', 'n_features'],
    ['ordinal', 'Unknown Value', 'unknown_value'],
    ['label', 'Missing/Unknown Code', 'missing_code'],
  ] as const)('preserves zero and the last value when clearing %s numeric options', (method, name, field) => {
    // Mid-edit empty numeric inputs must preserve the existing safe-parser fallback.
    const { onChange } = renderSettings({ method, [field]: 0 });
    const input = screen.getByRole('spinbutton', { name });
    expect(input).toHaveValue(0);
    fireEvent.change(input, { target: { value: '12.9' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: 12 }));
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ [field]: 12 }));
  });

  it('keeps ordinal fallback display, error disabling and category order behavior', () => {
    // The one-hot ignore setting displays as encoded-value without changing the payload.
    const { onChange } = renderSettings({ method: 'ordinal', unknown_value: 0 });
    expect(screen.getByRole('combobox', { name: 'Handle Unknown' })).toHaveValue('use_encoded_value');
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole('combobox', { name: 'Handle Unknown' }), { target: { value: 'error' } });
    expect(screen.getByRole('spinbutton', { name: 'Unknown Value' })).toBeDisabled();
    fireEvent.change(screen.getByRole('textbox', { name: 'Category Order (optional)' }), { target: { value: 'low, medium, high' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ categories_order: 'low, medium, high', unknown_value: 0 }));
  });

  it('keeps smoothing auto, numeric and empty-string conversion and target type selection', () => {
    // Empty smoothing currently means zero while nonnumeric text returns to auto.
    const { onChange } = renderSettings({ method: 'target', smooth: 0 });
    const input = screen.getByRole('textbox', { name: 'Smoothing' });
    expect(input).toHaveValue('0');
    for (const [value, expected] of [['2.5', 2.5], ['auto', 'auto'], ['invalid', 'auto'], ['', 0]] as const) {
      fireEvent.change(input, { target: { value } });
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ smooth: expected }));
    }
    fireEvent.change(screen.getByRole('combobox', { name: 'Target Type' }), { target: { value: 'binary' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ target_type: 'binary', smooth: 0 }));
  });

  it('retains WOE regularization zero and its existing float/empty parsing', () => {
    // Extraction must not silently normalize the existing NaN result on clear.
    const { onChange } = renderSettings({ method: 'woe', regularization: 0 });
    const input = screen.getByRole('spinbutton', { name: 'Regularization' });
    expect(input).toHaveValue(0);
    fireEvent.change(input, { target: { value: '1.25' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ regularization: 1.25 }));
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ regularization: NaN }));
  });

  it('filters dropped columns from feature and target selectors while preserving the WOE target list', () => {
    // WOE currently lists the whole schema; target encoding uses only remaining columns.
    const { onChange } = renderSettings({ method: 'target' });
    const picker = screen.getByRole('group', { name: 'Columns to Encode' });
    expect(within(picker).queryByRole('checkbox', { name: 'removed' })).not.toBeInTheDocument();
    fireEvent.click(within(picker).getByRole('checkbox', { name: 'city' }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ columns: ['city'] }));
    let target = screen.getByRole('combobox', { name: 'Target Column' });
    expect(within(target).queryByRole('option', { name: 'removed (object)' })).not.toBeInTheDocument();
    expect(target.closest('[data-validation-field]')).toHaveAttribute('data-validation-field', 'target_column');
    fireEvent.change(target, { target: { value: 'target' } });
    selectMethod('woe');
    target = screen.getByRole('combobox', { name: 'Target Column' });
    expect(target).toHaveValue('target');
    expect(within(target).getByRole('option', { name: 'removed (object)' })).toBeInTheDocument();
    expect(picker.closest('[data-validation-field]')).toHaveAttribute('data-validation-field', 'columns');
  });

  it('shows matching recommendations and wrapped metrics including zero counts', () => {
    // Feedback must retain backend metric unwrapping and all three recommendation filters.
    useGraphStore.setState({ executionResult: {
      pipeline_id: 'pipeline', status: 'success', preview_data: null,
      node_results: { encoder: { metrics: { steps: { encoding: { details: {
        encoded_columns_count: 0, new_features_count: 0, categories_count: { city: 3 }, classes_count: { target: 2 },
      } } } } } },
      recommendations: [
        ['node', 'other', 'encoding'], ['type', 'target_encoding', 'other'],
        ['cardinality', 'high_cardinality', 'other'], ['unrelated', 'scale', 'scaling'],
      ].map(([id, type, node]) => ({
        rule_id: id!, type: type!, suggested_node_type: node!, description: `${id} advice`,
        target_columns: [], suggested_params: {}, confidence: 1, reasoning: '',
      })),
    } });
    renderSettings();
    expect(screen.getByText('node advice')).toBeInTheDocument();
    expect(screen.getByText('type advice')).toBeInTheDocument();
    expect(screen.getByText('cardinality advice')).toBeInTheDocument();
    expect(screen.queryByText('unrelated advice')).not.toBeInTheDocument();
    expect(screen.getByText('Columns Encoded:').parentElement).toHaveTextContent('Columns Encoded:0');
    expect(screen.getByText('New Features Created:').parentElement).toHaveTextContent('New Features Created:0');
    expect(screen.getByText('Categories Found:').parentElement).toHaveTextContent('city:3');
    expect(screen.getByText('Classes Found:').parentElement).toHaveTextContent('target:2');
  });

  it('shows disconnected and loading notices only in their current states', () => {
    // A disabled schema query should not show a loading notice without a dataset.
    sources.datasetId = undefined;
    sources.isLoading = true;
    const { rerender } = render(<Settings config={EncodingNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.getByText('Connect a dataset node to see available columns.')).toBeInTheDocument();
    expect(screen.queryByText('Loading schema...')).not.toBeInTheDocument();
    sources.datasetId = 'dataset-1';
    rerender(<Settings config={EncodingNode.getDefaultConfig()} onChange={() => {}} />);
    expect(screen.queryByText('Connect a dataset node to see available columns.')).not.toBeInTheDocument();
    expect(screen.getByText('Loading schema...')).toBeInTheDocument();
  });

  it('preserves validation priority and implicit label/ordinal target behavior', () => {
    // Validation field IDs drive inspector navigation and target-safe empty selections.
    const config = EncodingNode.getDefaultConfig();
    expect(EncodingNode.validate({ ...config, method: 'label' })).toEqual({ isValid: true });
    expect(EncodingNode.validate({ ...config, method: 'ordinal' })).toEqual({ isValid: true });
    expect(EncodingNode.validate({ ...config, method: 'woe' })).toEqual({
      isValid: false, field: 'columns', message: 'Select at least one column',
    });
    expect(EncodingNode.validate({ ...config, method: 'woe', columns: ['city'] })).toEqual({
      isValid: false, field: 'target_column', message: 'WOE encoding requires a binary target column',
    });
    expect(EncodingNode.validate({ ...config, method: 'target', columns: ['city'] })).toEqual({ isValid: true });
  });
});
