import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { NodeDefinition } from '../../../core/types/nodes';
import { InvalidValueReplacementNode } from './InvalidValueReplacementNode';
import { DropColumnsNode } from './DropColumnsNode';
import { BinningNode } from './BinningNode';
import { CastTypeNode } from './CastTypeNode';
import { AliasReplacementNode } from './AliasReplacementNode';
import { MissingIndicatorNode } from './MissingIndicatorNode';
import { PolynomialFeaturesNode } from './PolynomialFeaturesNode';
import { DeduplicationNode } from './DeduplicationNode';
import { DropRowsNode } from './DropRowsNode';
import { FeatureInteractionNode } from './FeatureInteractionNode';

const fixtures = vi.hoisted(() => ({
  upstream: [{ datasetId: '' }, { datasetId: 'first-dataset' }, { datasetId: 'second-dataset' }],
  schema: undefined as { columns: Record<string, { name: string; dtype: string }> } | undefined,
  isLoading: false,
  dropped: new Set<string>(),
  wide: false,
  metrics: null as Record<string, unknown> | null,
  recommendations: [] as { rule_id: string; type: string; description: string; target_columns?: string[] }[],
  schemaRequest: vi.fn(),
  upstreamRequest: vi.fn(),
  recommendationsRequest: vi.fn(),
}));
vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: (id: string) => {
  fixtures.upstreamRequest(id);
  return fixtures.upstream;
} }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: (id: string) => {
  fixtures.schemaRequest(id);
  return { data: fixtures.schema, isLoading: fixtures.isLoading };
} }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => fixtures.dropped }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [React.createRef<HTMLDivElement>(), fixtures.wide] }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: (...args: unknown[]) => {
  fixtures.recommendationsRequest(...args);
  return fixtures.recommendations;
} }));
vi.mock('../../../core/store/useGraphStore', () => ({ useGraphStore: (selector: (state: unknown) => unknown) => selector({
  executionResult: { node_results: { node: { metrics: fixtures.metrics }, other: { metrics: { dropped_columns_count: 999 } } } },
}) }));

const nodes: NodeDefinition[] = [InvalidValueReplacementNode, DropColumnsNode, BinningNode, CastTypeNode,
  AliasReplacementNode, MissingIndicatorNode, PolynomialFeaturesNode, DeduplicationNode, DropRowsNode, FeatureInteractionNode];

function mount(definition: NodeDefinition, overrides: Record<string, unknown> = {}) {
  let config = { ...definition.getDefaultConfig(), ...overrides };
  const onChange = vi.fn();
  const Settings = definition.settings;
  const view = render(<Settings config={config} onChange={onChange} nodeId="node" />);
  return { ...view, onChange, config, update: (updates: Record<string, unknown> = {}) => {
    config = { ...config, ...updates };
    view.rerender(<Settings config={config} onChange={onChange} nodeId="node" />);
  } };
}

beforeEach(() => {
  vi.clearAllMocks();
  fixtures.upstream = [{ datasetId: '' }, { datasetId: 'first-dataset' }, { datasetId: 'second-dataset' }];
  fixtures.schema = { columns: Object.fromEntries([
    ['age', 'Int64'], ['city', 'object'], ['score', 'FLOAT32'], ['removed', 'number'],
    ['wide', 'double'], ['long', 'long'], ['flag', 'bool'], ['category', 'category'], ['text', 'TEXT'], ['string', 'string'],
  ].map(([name, dtype]) => [name, { name: name!, dtype: dtype! }])) };
  fixtures.dropped = new Set(['removed']);
  fixtures.isLoading = false;
  fixtures.wide = false;
  fixtures.metrics = null;
  fixtures.recommendations = [];
});

describe('preprocessing public settings contracts', () => {
  it.each(nodes)('$type uses the first nonempty dataset and preserves config on resize', (definition) => {
    // Settings must read their own upstream input without changing the pipeline on resize.
    const view = mount(definition);
    expect(fixtures.schemaRequest).toHaveBeenLastCalledWith('first-dataset');
    expect(fixtures.upstreamRequest).toHaveBeenLastCalledWith('node');
    fixtures.wide = true;
    view.update();
    expect(view.onChange).not.toHaveBeenCalled();
    expect(view.container.firstElementChild).toBeTruthy();
  });

  const selections = [
    { definition: InvalidValueReplacementNode, label: 'Target Columns', expected: ['age', 'score'], field: 'columns' },
    { definition: BinningNode, label: 'Target Columns', expected: ['age', 'score'], field: 'columns' },
    { definition: AliasReplacementNode, label: 'Target Columns', expected: ['city', 'category', 'text', 'string'], field: 'columns' },
    { definition: PolynomialFeaturesNode, label: 'Input Columns (Numeric)', expected: ['age', 'score', 'wide', 'long'], field: 'columns' },
    { definition: FeatureInteractionNode, label: 'Input Columns (Numeric)', expected: ['age', 'score', 'wide', 'long'], field: 'columns' },
    ...[
      { definition: DropColumnsNode, label: 'Explicitly Drop Columns', field: 'columns' },
      { definition: MissingIndicatorNode, label: 'Target Columns (Optional)', field: 'columns' },
      { definition: DeduplicationNode, label: 'Subset Columns (Optional)', field: 'subset' },
    ].map(item => ({ ...item, expected: ['age', 'city', 'score', 'wide', 'long', 'flag', 'category', 'text', 'string'] })),
  ];
  it.each(selections)('$definition.type filters columns in schema order and preserves hidden selections', ({ definition, label, expected, field }) => {
    // Dtype and upstream drops must not discard configured columns absent from the current picker.
    const view = mount(definition, { [field]: ['hidden'] });
    const group = screen.getByRole('group', { name: label });
    expect(within(group).getAllByRole('checkbox').map(input => input.closest('label')?.textContent?.trim())).toEqual(expected);
    fireEvent.click(within(group).getByRole('checkbox', { name: expected[0]! }));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, [field]: ['hidden', expected[0]] });
    view.update({ [field]: ['hidden', expected[0]] });
    fixtures.wide = true;
    view.update();
    expect(within(group).getByRole('checkbox', { name: expected[0]! })).toBeChecked();
    expect(fixtures.schemaRequest).toHaveBeenLastCalledWith('first-dataset');
  });
  it.each(selections)('$definition.type tolerates missing schema', ({ definition, label }) => {
    // Disconnected and loading nodes must expose an empty picker without emitting changes.
    fixtures.schema = undefined;
    fixtures.upstream = [];
    const view = mount(definition);
    expect(within(screen.getByRole('group', { name: label })).queryAllByRole('checkbox')).toHaveLength(0);
    expect(fixtures.schemaRequest).toHaveBeenLastCalledWith(undefined);
    expect(view.onChange).not.toHaveBeenCalled();
  });

  it.each(['negative_to_nan', 'zero_to_nan', 'percentage_bounds', 'age_bounds', 'custom_range'])('preserves invalid-value %s mode and range values', (mode) => {
    // Switching presets must retain custom bounds, including zero, for later edits.
    const view = mount(InvalidValueReplacementNode, { columns: ['age'], min_value: 0, max_value: 120 });
    fireEvent.change(screen.getByLabelText('Replacement Mode'), { target: { value: mode } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, mode });
    view.update({ mode });
    expect(screen.queryByLabelText('Min Value') !== null).toBe(['percentage_bounds', 'age_bounds', 'custom_range'].includes(mode));
  });
  it('keeps range zero distinct from cleared values and preserves info state on resize', () => {
    // Clearing an optional bound means undefined; numeric zero remains configured.
    const view = mount(InvalidValueReplacementNode, { mode: 'custom_range', min_value: 3, max_value: 10 });
    fireEvent.change(screen.getByLabelText('Min Value'), { target: { value: '0' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, min_value: 0 });
    fireEvent.change(screen.getByLabelText('Max Value'), { target: { value: '' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, max_value: undefined });
    fireEvent.click(screen.getByRole('button', { name: 'About Invalid Value Replacement' }));
    fixtures.wide = true;
    view.update();
    expect(screen.queryByText(/Use this node to handle data quality/)).toBeNull();
  });
  it('preserves bin edge drafts on rerender and parses sorted numeric values only on blur', () => {
    // Parent rerenders must not erase a draft unless the externally configured edges change.
    const view = mount(BinningNode, { strategy: 'custom', columns: ['age'], custom_bins: { age: [0, 10], hidden: [3, 4] } });
    const edges = screen.getByLabelText('Bin edges for age');
    fireEvent.change(edges, { target: { value: '20, nope, 0, -1, 10x' } });
    fixtures.wide = true;
    view.update({ custom_bins: { age: [0, 10], hidden: [3, 4] } });
    expect(edges).toHaveValue('20, nope, 0, -1, 10x');
    expect(view.onChange).not.toHaveBeenCalled();
    fireEvent.blur(edges);
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, custom_bins: { age: [-1, 0, 10, 20], hidden: [3, 4] } });
    view.update({ custom_bins: { age: [1, 5] } });
    expect(edges).toHaveValue('1, 5');
  });
  it('keeps bin count fallbacks, precision zero and hidden suffix semantics', () => {
    // The existing numeric fallbacks and retained hidden options are part of saved configurations.
    const view = mount(BinningNode, { label_format: 'range', precision: 2, n_bins: 8, output_suffix: '_saved' });
    for (const value of ['0', '']) {
      fireEvent.change(screen.getByLabelText('Number of Bins'), { target: { value } });
      expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, n_bins: 5 });
      fireEvent.change(screen.getByLabelText('Precision (Decimals)'), { target: { value } });
      expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, precision: 0 });
    }
    fireEvent.click(screen.getByLabelText('Drop Original Columns'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, drop_original: true });
    view.update({ drop_original: true });
    expect(screen.queryByLabelText('Output Suffix')).toBeNull();
    view.update({ drop_original: false });
    expect(screen.getByLabelText('Output Suffix')).toHaveValue('_saved');
  });
  it('adds, renames, changes and removes cast rules in available column order', () => {
    // Rule edits must retain unrelated entries and exclude upstream dropped columns.
    const view = mount(CastTypeNode, { column_types: { age: 'int', hidden: 'bool' } });
    fireEvent.click(screen.getByLabelText('Add Casting Rule'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, column_types: { age: 'int', hidden: 'bool', city: 'float' } });
    expect(within(screen.getByLabelText('Column for casting rule age')).queryByRole('option', { name: 'removed' })).toBeNull();
    fireEvent.change(screen.getByLabelText('Column for casting rule age'), { target: { value: 'score' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, column_types: { hidden: 'bool', score: 'int' } });
    fireEvent.change(screen.getByLabelText('Data type for age'), { target: { value: 'float32' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, column_types: { age: 'float32', hidden: 'bool' } });
    fireEvent.click(screen.getByLabelText('Remove casting rule for age'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, column_types: { hidden: 'bool' } });
  });
  it('preserves the existing cast add fallback when every column has a rule', () => {
    // The original add action overwrites the first column when all options are already assigned.
    fixtures.schema = { columns: { age: { name: 'age', dtype: 'int' } } };
    const view = mount(CastTypeNode, { column_types: { age: 'int' } });
    fireEvent.click(screen.getByLabelText('Add Casting Rule'));
    expect(view.onChange).toHaveBeenLastCalledWith({ column_types: { age: 'float' } });
    fixtures.schema = undefined;
    view.update();
    expect(screen.getByLabelText('Add Casting Rule')).toBeDisabled();
  });
  it('preserves alias drafts on resize and retains custom mappings across modes', () => {
    // Local drafts and configured mappings have different lifetimes and must not be conflated.
    const view = mount(AliasReplacementNode, { columns: ['city'], custom_pairs: { NYC: 'New York' } });
    expect(screen.getByLabelText('Add alias mapping')).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Alias (Old)'), { target: { value: ' NY ' } });
    fireEvent.change(screen.getByLabelText('Canonical (New)'), { target: { value: ' New York ' } });
    fixtures.wide = true;
    view.update();
    expect(screen.getByLabelText('Alias (Old)')).toHaveValue(' NY ');
    fireEvent.click(screen.getByLabelText('Add alias mapping'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, custom_pairs: { NYC: 'New York', ' NY ': ' New York ' } });
    expect(screen.getByLabelText('Alias (Old)')).toHaveValue('');
    fireEvent.click(screen.getByLabelText('Remove alias NYC'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, custom_pairs: {} });
    fireEvent.change(screen.getByLabelText('Replacement Mode'), { target: { value: 'normalize_boolean' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, mode: 'normalize_boolean' });
    view.update({ mode: 'normalize_boolean' });
    expect(screen.queryByLabelText('Alias (Old)')).toBeNull();
    view.update({ mode: 'custom' });
    expect(screen.getByLabelText('Remove alias NYC')).toBeVisible();
  });
  it('merges recommendations in order and preserves the threshold', () => {
    // Applying recommendations must deduplicate without replacing explicit selections.
    fixtures.recommendations = [{ rule_id: 'drop', type: 'cleaning', description: 'Drop empty', target_columns: ['age', 'score', 'city'] }];
    const view = mount(DropColumnsNode, { columns: ['city', 'age'], missing_threshold: 35 });
    fireEvent.click(screen.getByTitle('Apply Recommendation'));
    expect(view.onChange).toHaveBeenLastCalledWith({ columns: ['city', 'age', 'score'], missing_threshold: 35 });
    expect(fixtures.recommendationsRequest).toHaveBeenLastCalledWith('node', {
      types: ['cleaning', 'feature_selection'], suggestedNodeTypes: ['DropMissingColumns'], scope: 'column',
    });
    fireEvent.change(screen.getByLabelText('Missing Value Threshold'), { target: { value: '0' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, missing_threshold: 0 });
  });
  it('preserves row threshold while toggling any-missing and accepts zero', () => {
    // The threshold stays configured while its parent section is visually disabled.
    const view = mount(DropRowsNode, { missing_threshold: 35 });
    fireEvent.click(screen.getByLabelText('Drop rows with ANY missing values'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, drop_if_any_missing: true });
    view.update({ drop_if_any_missing: true });
    expect(screen.getByLabelText('Missing Value Threshold (%)')).toHaveValue('35');
    fireEvent.change(screen.getByLabelText('Missing Value Threshold (%)'), { target: { value: '0' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, drop_if_any_missing: true, missing_threshold: 0 });
  });
  it('keeps empty indicator suffixes and duplicate keep choices', () => {
    // Free text is passed through exactly and keep-none is a string enum rather than false.
    const indicator = mount(MissingIndicatorNode);
    fireEvent.change(screen.getByLabelText('Indicator Suffix'), { target: { value: '' } });
    expect(indicator.onChange).toHaveBeenLastCalledWith({ ...indicator.config, flag_suffix: '' });
    indicator.unmount();
    const deduplicate = mount(DeduplicationNode, { subset: ['age'] });
    fireEvent.change(screen.getByLabelText('Keep Strategy'), { target: { value: 'none' } });
    expect(deduplicate.onChange).toHaveBeenLastCalledWith({ ...deduplicate.config, keep: 'none' });
  });
  it.each([PolynomialFeaturesNode, FeatureInteractionNode])('$type preserves degree and checkbox payloads across collapse', (definition) => {
    // Collapsing settings must leave hidden feature parameters available when reopened.
    const view = mount(definition, { columns: ['age', 'score'], degree: 3 });
    fireEvent.change(screen.getByLabelText('Degree'), { target: { value: '4' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, degree: 4 });
    fireEvent.click(screen.getByLabelText('Include Bias'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, include_bias: true });
    fireEvent.click(screen.getByLabelText('Interaction Only'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, interaction_only: !view.config.interaction_only });
    fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, isExpanded: false });
    view.update({ isExpanded: false });
    expect(screen.queryByLabelText('Degree')).toBeNull();
    fixtures.wide = true;
    view.update({ isExpanded: true });
    expect(screen.getByLabelText('Degree')).toHaveValue(definition === PolynomialFeaturesNode ? 3 : '3');
  });
  it('retains polynomial empty and zero display fallbacks', () => {
    // Persisted zero and empty prefix currently display defaults while edits emit the raw prefix.
    const view = mount(PolynomialFeaturesNode, { degree: 0, output_prefix: '' });
    expect(screen.getByLabelText('Degree')).toHaveValue(2);
    expect(screen.getByLabelText('Output Prefix')).toHaveValue('poly');
    fireEvent.change(screen.getByLabelText('Degree'), { target: { value: '' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, degree: 2 });
    fireEvent.change(screen.getByLabelText('Output Prefix'), { target: { value: '' } });
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, output_prefix: '' });
    fireEvent.click(screen.getByLabelText('Include Input Features'));
    expect(view.onChange).toHaveBeenLastCalledWith({ ...view.config, include_input_features: true });
  });

  it.each([
    { definition: DropColumnsNode, details: { dropped_columns_count: 0, dropped_columns: ['age'] }, labels: ['Columns Dropped:', 'Dropped Names:', 'age'] },
    { definition: CastTypeNode, details: { cast_errors: 0, casted_columns_count: 0 }, labels: ['Columns Casted:', 'Errors Encountered:'] },
    { definition: MissingIndicatorNode, details: { missing_indicators_created: 0, missing_indicators_columns: ['new_flag'] }, labels: ['Indicators Created:', 'New Columns:', 'new_flag'] },
    { definition: DeduplicationNode, details: { Deduplicate_rows_removed: 0, Deduplicate_rows_remaining: 10, Deduplicate_rows_total: 10 }, labels: ['Duplicates Removed:', 'Rows Remaining:', 'Total Rows:'] },
    { definition: DropRowsNode, details: { DropMissingRows_rows_removed: 0, DropMissingRows_rows_remaining: 10, DropMissingRows_rows_total: 10 }, labels: ['Rows Removed:', 'Rows Remaining:', 'Total Rows:'] },
  ])('$definition.type renders zero-valued wrapped results from its own node', ({ definition, details, labels }) => {
    // Zero-valued results must be visible and unrelated node metrics must not bleed into feedback.
    fixtures.metrics = { steps: { only: { details } } };
    const view = mount(definition);
    expect(screen.getByText('Last Run Results')).toBeVisible();
    for (const label of labels) expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    expect(screen.getAllByText('0').length).toBeGreaterThan(0);
    expect(screen.queryByText('999')).toBeNull();
    fixtures.metrics = {};
    view.update();
    expect(screen.queryByText('Last Run Results')).toBeNull();
  });

  it('retains all definition defaults, handles, validation and summaries', () => {
    // Public definitions are serialized separately from settings and must remain compatible.
    expect(nodes.map(definition => {
      const defaults = definition.getDefaultConfig();
      const configured = { ...defaults, columns: ['age', 'score'], subset: ['age'], column_types: { age: 'int' },
        min_value: 0, max_value: 10, custom_pairs: { NY: 'New York' }, missing_threshold: 25 };
      return { type: definition.type, inputs: definition.inputs, outputs: definition.outputs, defaults,
        defaultValidation: definition.validate(defaults), configuredValidation: definition.validate(configured),
        defaultSummary: definition.bodyPreview?.(defaults), configuredSummary: definition.bodyPreview?.(configured) };
    })).toMatchInlineSnapshot(`
      [
        {
          "configuredSummary": "negative_to_nan · 2 cols",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "negative_to_nan",
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select at least one column.",
          },
          "defaults": {
            "columns": [],
            "mode": "negative_to_nan",
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Cleaned Data",
              "type": "dataset",
            },
          ],
          "type": "InvalidValueReplacement",
        },
        {
          "configuredSummary": "Drop 2 cols · missing > 25%",
          "configuredValidation": {
            "field": "columns",
            "isValid": true,
            "message": undefined,
          },
          "defaultSummary": null,
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select columns or set a threshold",
          },
          "defaults": {
            "columns": [],
            "missing_threshold": 0,
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "type": "drop_missing_columns",
        },
        {
          "configuredSummary": "equal_width · q=5 · 2 cols",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "equal_width · q=5",
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select at least one column to bin.",
          },
          "defaults": {
            "columns": [],
            "drop_original": false,
            "label_format": "ordinal",
            "n_bins": 5,
            "output_suffix": "_binned",
            "strategy": "equal_width",
          },
          "inputs": [
            {
              "id": "in",
              "label": "Dataset",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Binned",
              "type": "dataset",
            },
          ],
          "type": "BinningNode",
        },
        {
          "configuredSummary": "Cast 1 col",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": null,
          "defaultValidation": {
            "isValid": true,
          },
          "defaults": {
            "column_types": {},
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Casted Data",
              "type": "dataset",
            },
          ],
          "type": "casting",
        },
        {
          "configuredSummary": "custom · 2 cols",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "custom",
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select at least one column.",
          },
          "defaults": {
            "columns": [],
            "custom_pairs": {},
            "mode": "custom",
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Standardized Data",
              "type": "dataset",
            },
          ],
          "type": "AliasReplacement",
        },
        {
          "configuredSummary": "2 cols → flags",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": null,
          "defaultValidation": {
            "isValid": true,
          },
          "defaults": {
            "columns": [],
            "flag_suffix": "_was_missing",
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Augmented Data",
              "type": "dataset",
            },
          ],
          "type": "MissingIndicator",
        },
        {
          "configuredSummary": "degree=2 · 2 cols",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "degree=2",
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select at least one input column.",
          },
          "defaults": {
            "columns": [],
            "degree": 2,
            "include_bias": false,
            "include_input_features": false,
            "interaction_only": false,
            "isExpanded": true,
            "output_prefix": "poly",
          },
          "inputs": [
            {
              "id": "in",
              "label": "Input Dataset",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Transformed Data",
              "type": "dataset",
            },
          ],
          "type": "PolynomialFeaturesNode",
        },
        {
          "configuredSummary": "Subset: 1 col · keep first",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "Subset: all · keep first",
          "defaultValidation": {
            "isValid": true,
          },
          "defaults": {
            "keep": "first",
            "subset": [],
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Unique Data",
              "type": "dataset",
            },
          ],
          "type": "deduplicate",
        },
        {
          "configuredSummary": "Drop rows missing > 25%",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "Drop rows missing > 50%",
          "defaultValidation": {
            "isValid": true,
          },
          "defaults": {
            "drop_if_any_missing": false,
            "missing_threshold": 50,
          },
          "inputs": [
            {
              "id": "in",
              "label": "Data",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Cleaned Data",
              "type": "dataset",
            },
          ],
          "type": "drop_missing_rows",
        },
        {
          "configuredSummary": "degree=2 · 2 cols",
          "configuredValidation": {
            "isValid": true,
          },
          "defaultSummary": "degree=2",
          "defaultValidation": {
            "field": "columns",
            "isValid": false,
            "message": "Select at least one input column.",
          },
          "defaults": {
            "columns": [],
            "degree": 2,
            "include_bias": false,
            "interaction_only": true,
            "isExpanded": true,
          },
          "inputs": [
            {
              "id": "in",
              "label": "Input Dataset",
              "type": "dataset",
            },
          ],
          "outputs": [
            {
              "id": "out",
              "label": "Transformed Data",
              "type": "dataset",
            },
          ],
          "type": "FeatureInteractionNode",
        },
      ]
    `);
  });
});
