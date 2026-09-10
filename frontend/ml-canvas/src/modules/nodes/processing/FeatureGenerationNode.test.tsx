import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import type { NodeSettingsProps } from '../../../core/types/nodes';
import { initializeRegistry } from '../../../core/registry/init';
import { ValidationNavigation } from '../../../components/shared/ValidationField';
import { useViewStore } from '../../../core/store/useViewStore';
import { FeatureGenerationNode } from './FeatureGenerationNode';

type FeatureGenerationConfig = ReturnType<typeof FeatureGenerationNode.getDefaultConfig>;

const fixture = { wide: false, dropped: new Set<string>(), columns: {
  age: { name: 'age', dtype: 'int' }, income: { name: 'income', dtype: 'FLOAT64' },
  signup_date: { name: 'signup_date', dtype: 'datetime' }, city: { name: 'city', dtype: 'string' },
  name: { name: 'name', dtype: 'object' },
} as Record<string, { name: string; dtype: string }> };

const mockGraphState = {
  nodes: [],
  edges: [],
  executionResult: {
    node_results: {} as Record<string, { metrics: unknown }>,
  },
};

const recommendationFixture = [
  {
    rule_id: 'fg-001',
    type: 'feature_generation',
    target_columns: ['age', 'income'],
    description: 'Create a feature from the selected columns.',
    suggested_node_type: 'FeatureGenerationNode',
    suggested_params: {
      columns: ['age', 'income'],
    },
    confidence: 0.94,
    reasoning: 'These columns are frequently combined in downstream models.',
  },
];

vi.mock('../../../core/hooks/useUpstreamData', () => ({
  useUpstreamData: () => [{ datasetId: 'ds-1' }],
}));

vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: () => ({
    data: {
      columns: fixture.columns,
    },
    isLoading: false,
  }),
}));

vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => fixture.dropped,
}));

vi.mock('../../../core/hooks/useRecommendations', () => ({
  useRecommendations: () => recommendationFixture,
}));

vi.mock('../../../core/store/useGraphStore', () => ({
  collectGraphValidationIssues: () => [],
  useGraphStore: (selector: (state: typeof mockGraphState) => unknown) => selector(mockGraphState),
}));

vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), fixture.wide] as const,
}));

function renderFeatureGeneration(config: FeatureGenerationConfig, onChange = vi.fn()) {
  const client = new QueryClient();
  const Settings = FeatureGenerationNode.settings as React.JSXElementConstructor<NodeSettingsProps<FeatureGenerationConfig>>;

  const view = render(
    <QueryClientProvider client={client}>
      <Settings config={config} onChange={onChange} nodeId="feature-node" />
    </QueryClientProvider>,
  );

  return { onChange, ...view };
}

describe('FeatureGenerationNode recommendations', () => {
  beforeAll(() => initializeRegistry());

  beforeEach(() => {
    fixture.wide = false;
    fixture.dropped = new Set();
    useViewStore.setState({ validationFocusRequest: null });
    mockGraphState.nodes = [];
    mockGraphState.edges = [];
    mockGraphState.executionResult = { node_results: {} };
  });

  it('keeps feature-generation recommendations informational and hides the Apply action', () => {
    // Recommendations remain advice and must never mutate the operation configuration.
    const config: FeatureGenerationConfig = {
      operations: [
        {
          operation_type: 'arithmetic',
          method: 'add',
          input_columns: ['age'],
          secondary_columns: ['income'],
          output_column: 'age_plus_income',
          isExpanded: true,
        },
      ],
    };

    const { onChange } = renderFeatureGeneration(config);

    fireEvent.click(screen.getByRole('button', { name: /recommendations \(1\)/i }));

    expect(screen.getByText('Create a feature from the selected columns.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /apply recommendation/i })).not.toBeInTheDocument();
    expect(screen.getByDisplayValue('age_plus_income')).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});

function operation(operation_type = 'arithmetic', extra = {}) {
  return { operation_type, method: 'add', input_columns: [], output_column: '', isExpanded: true, ...extra };
}

function renderControlled(operations = [operation()]) {
  const onChange = vi.fn();
  const Settings = FeatureGenerationNode.settings;
  function Harness() {
    const [config, setConfig] = React.useState({ operations });
    return <ValidationNavigation nodeId="feature-node"><Settings config={config} nodeId="feature-node" onChange={(next) => {
      onChange(next); setConfig(next);
    }} /></ValidationNavigation>;
  }
  const view = render(<Harness />);
  return { onChange, ...view };
}

function choose(group: string, column: string) {
  fireEvent.click(within(screen.getByRole('group', { name: group })).getByRole('checkbox', { name: column }));
}

describe('FeatureGenerationNode public contracts', () => {
  beforeEach(() => {
    fixture.wide = false;
    fixture.dropped = new Set();
    mockGraphState.executionResult = { node_results: {} };
    useViewStore.setState({ validationFocusRequest: null });
  });

  it.each([
    ['Arithmetic', 'arithmetic', 'add', undefined],
    ['Date Extraction', 'datetime_extract', 'year', ['year']],
    ['Ratio', 'ratio', 'year', undefined],
    ['Similarity', 'similarity', 'ratio', undefined],
    ['Group Agg', 'group_agg', 'mean', undefined],
  ])('adds %s with its existing defaults', (label, operation_type, method, datetime_features) => {
    // Saved payload defaults must remain stable even where a method is unused by that operation.
    const { onChange } = renderFeatureGeneration({});
    fireEvent.click(screen.getByTitle(label as string));
    expect(onChange).toHaveBeenLastCalledWith({ operations: [{ operation_type, method, input_columns: [], output_column: '', datetime_features, isExpanded: true }] });
  });

  it.each([false, true])('preserves add controls in wide=%s layout', (wide) => {
    // Width changes presentation without changing the five available operation choices.
    fixture.wide = wide;
    const { container } = renderFeatureGeneration({ operations: [] });
    expect(container.firstChild).toHaveClass(wide ? 'flex' : 'flex-col');
    expect(screen.getByText(wide ? 'Add' : 'Add:')).toBeVisible();
    expect(screen.getByText('No operations added')).toBeVisible();
    expect(screen.getAllByTitle(/Arithmetic|Date Extraction|Ratio|Similarity|Group Agg/)).toHaveLength(5);
  });

  it.each([
    ['arithmetic', 'subtract', ['add', 'subtract', 'multiply', 'divide']],
    ['similarity', 'token_set_ratio', ['ratio', 'token_sort_ratio', 'token_set_ratio']],
    ['group_agg', 'median', ['mean', 'sum', 'count', 'min', 'max', 'std', 'median']],
  ])('updates the %s method and keeps expansion and operands', (type, method, methods) => {
    // Header method updates must merge with the current operation rather than resetting it.
    const original = operation(type as string, { input_columns: ['age'], constants: [0], isExpanded: false });
    const { onChange } = renderControlled([original]);
    const select = screen.getByRole('combobox', { name: 'Method for operation 1' });
    expect(within(select).getAllByRole('option').map(o => o.getAttribute('value'))).toEqual(methods);
    fireEvent.change(select, { target: { value: method } });
    expect(onChange).toHaveBeenLastCalledWith({ operations: [{ ...original, method }] });
    expect(screen.getByRole('button', { name: 'Expand ' + type.toString().replace('_', ' ') + ' operation 1' })).toHaveAttribute('aria-expanded', 'false');
  });

  it.each([
    ['arithmetic', 'Column A (Left Operand)', 'Column B (Right Operand)', 'age', 'income'],
    ['similarity', 'String A', 'String B', 'city', 'name'],
    ['group_agg', 'Group By (Categorical)', 'Target (Numeric)', 'city', 'income'],
    ['ratio', 'Numerator (Sum)', 'Denominator (Sum)', 'age', 'income'],
  ])('updates %s operands through the actual column pickers', (type, left, right, first, second) => {
    // Each editor must write the correct operand fields and preserve imported constants.
    const { onChange } = renderControlled([operation(type, { constants: [3] })]);
    choose(`${left} for operation 1`, first);
    choose(`${right} for operation 1`, second);
    expect(onChange).toHaveBeenLastCalledWith({ operations: [operation(type, { input_columns: [first], secondary_columns: [second], constants: [3] })] });
  });

  it('keeps ratio operands multiple and arithmetic operands single', () => {
    // Ratio aggregates multiple columns while arithmetic replaces a single operand.
    const { onChange } = renderControlled([operation('ratio'), operation()]);
    choose('Numerator (Sum) for operation 1', 'age');
    choose('Numerator (Sum) for operation 1', 'income');
    choose('Column A (Left Operand) for operation 2', 'age');
    choose('Column A (Left Operand) for operation 2', 'income');
    expect(onChange.mock.lastCall?.[0].operations.map((op: { input_columns: string[] }) => op.input_columns)).toEqual([['age', 'income'], ['income']]);
  });

  it('filters dropped columns without clearing selected values', () => {
    // Upstream availability restricts choices but must not rewrite stored operands.
    fixture.dropped = new Set(['income']);
    const { onChange } = renderControlled([operation('arithmetic', { secondary_columns: ['income'] })]);
    expect(within(screen.getByRole('group', { name: 'Column B (Right Operand) for operation 1' })).queryByRole('checkbox', { name: 'income' })).not.toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: 'Column A (Left Operand) for operation 1' })).queryByRole('checkbox', { name: 'city' })).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('falls back to all available columns for dates and grouping', () => {
    // Untyped imports remain selectable when no date or categorical columns survive upstream.
    fixture.dropped = new Set(['city', 'name', 'signup_date']);
    renderControlled([operation('datetime_extract'), operation('group_agg')]);
    expect(within(screen.getByRole('group', { name: 'Date Column for operation 1' })).getAllByRole('checkbox').map(c => c.closest('label')?.textContent)).toEqual(['age', 'income']);
    expect(within(screen.getByRole('group', { name: 'Group By (Categorical) for operation 2' })).getByRole('checkbox', { name: 'age' })).toBeVisible();
  });

  it('updates date selections, feature checkboxes and output name', () => {
    // Date feature toggles preserve ordering and support the original empty feature fallback.
    const { onChange } = renderControlled([operation('datetime_extract')]);
    choose('Date Column for operation 1', 'signup_date');
    fireEvent.click(screen.getByRole('checkbox', { name: 'year int' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'month name string' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'year int' }));
    const output = screen.getByRole('textbox', { name: 'Output Column Name for operation 1' });
    expect(output).toHaveAttribute('placeholder', 'datetime_extract_0');
    fireEvent.change(output, { target: { value: 'calendar' } });
    expect(onChange).toHaveBeenLastCalledWith({ operations: [operation('datetime_extract', { input_columns: ['signup_date'], datetime_features: ['month_name'], output_column: 'calendar' })] });
    expect(screen.getByRole('checkbox', { name: 'month name string' }).closest('label')).toHaveAttribute('title', expect.stringContaining('output type: string'));
  });

  it('supports explicit collapse, expand and removal with changed indices', () => {
    // Removing a row must retain the following operation and derive its new output placeholder.
    const { onChange } = renderControlled([operation(), operation('ratio')]);
    fireEvent.click(screen.getByRole('button', { name: 'Collapse arithmetic operation 1' }));
    expect(screen.queryByRole('textbox', { name: 'Output Column Name for operation 1' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand arithmetic operation 1' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove operation 1' }));
    expect(screen.getByRole('textbox', { name: 'Output Column Name for operation 1' })).toHaveAttribute('placeholder', 'ratio_0');
    expect(onChange).toHaveBeenLastCalledWith({ operations: [operation('ratio')] });
  });

  it('reveals locally, shifts revealed indices on deletion and allows collapse', () => {
    // Validation navigation must not persist expansion into a saved pipeline.
    const { onChange } = renderControlled([operation('ratio', { isExpanded: false }), operation('arithmetic', { isExpanded: false })]);
    act(() => useViewStore.getState().requestValidationFocus({ nodeId: 'feature-node', nodeLabel: 'Feature Generation', category: 'configuration', field: 'operations.1.secondary_columns', message: 'Arithmetic requires two operands.' }));
    expect(screen.getByRole('button', { name: 'Collapse arithmetic operation 2' })).toHaveAttribute('aria-expanded', 'true');
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Remove operation 1' }));
    expect(screen.getByRole('button', { name: 'Collapse arithmetic operation 1' })).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Collapse arithmetic operation 1' }));
    expect(screen.getByRole('button', { name: 'Expand arithmetic operation 1' })).toHaveAttribute('aria-expanded', 'false');
    expect(onChange).toHaveBeenLastCalledWith({ operations: [operation('arithmetic', { isExpanded: false })] });
  });

  it.each([
    [{ generated_features: ['total', 42] }, ['total', '42']],
    [{ steps: { feature: { details: { generated_features: ['wrapped'] } } } }, ['wrapped']],
    [{ generated_features: 'invalid' }, []],
    [{}, []],
  ])('renders supported execution feedback %j', (metrics, generated) => {
    // Both raw and wrapped metrics must display names with the existing string conversion.
    mockGraphState.executionResult = { node_results: { 'feature-node': { metrics } } };
    renderFeatureGeneration({ operations: [] });
    expect(screen.getByText('Last Run Results')).toBeVisible();
    for (const name of generated) expect(screen.getByText(name)).toBeVisible();
    expect(Boolean(screen.queryByText('No features generated in last run.'))).toBe(generated.length === 0);
  });

  it.each([
    [[], 'operations', 'Add at least one operation.'],
    [[operation()], 'operations.0.input_columns', 'Arithmetic requires two operands.'],
    [[operation('arithmetic', { input_columns: ['age'] })], 'operations.0.secondary_columns', 'Arithmetic requires two operands.'],
    [[operation('datetime_extract')], 'operations.0.input_columns', 'Select a date column.'],
    [[operation('group_agg')], 'operations.0.input_columns', 'Select both Group By and Target columns.'],
    [[operation('group_agg', { input_columns: ['city'] })], 'operations.0.secondary_columns', 'Select both Group By and Target columns.'],
  ])('preserves validation field and message for %j', (operations, field, message) => {
    // Validation ordering and exact messages drive the public Problems navigation contract.
    expect(FeatureGenerationNode.validate({ operations })).toEqual({ isValid: false, field, message });
  });

  it('preserves imported constants, permissive ratio/similarity validation and preview', () => {
    // The extraction must preserve current validators rather than add new checks to unused fields.
    const operations = [operation('arithmetic', { input_columns: ['age'], constants: [0] }), operation('ratio'), operation('similarity')];
    expect(FeatureGenerationNode.validate({ operations })).toEqual({ isValid: true });
    expect(FeatureGenerationNode.getDefaultConfig()).toEqual({ operations: [] });
    expect(FeatureGenerationNode.bodyPreview?.({})).toBeNull();
    expect(FeatureGenerationNode.bodyPreview?.({ operations: [operation()] })).toBe('+1 feature');
    expect(FeatureGenerationNode.bodyPreview?.({ operations })).toBe('+3 features');
    expect(FeatureGenerationNode.inputs).toEqual([{ id: 'in', type: 'dataset', label: 'Dataset' }]);
    expect(FeatureGenerationNode.outputs).toEqual([{ id: 'out', type: 'dataset', label: 'Enhanced' }]);
  });
});
