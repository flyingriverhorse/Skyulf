import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { NodeDefinition } from '../../../core/types/nodes';

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [{ datasetId: 'data' }] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: () => ({ data: { columns: {
    age: { name: 'age', dtype: 'int' },
    income: { name: 'income', dtype: 'float' },
    text: { name: 'text', dtype: 'string' },
    date: { name: 'date', dtype: 'datetime' },
  } }, isLoading: false }),
}));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set() }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => [] }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), false],
}));

const modules = import.meta.glob<Record<string, unknown>>(['./*Node.tsx', './*Nodes.tsx', './ValueReplacementNode.ts'], { eager: true });
const definitions = Object.values(modules).flatMap(module => Object.values(module)).filter(
  (value): value is NodeDefinition => typeof value === 'object' && value !== null && 'getDefaultConfig' in value,
);

const variants: Record<string, Record<string, unknown>[]> = {
  encoding: ['onehot', 'dummy', 'ordinal', 'target', 'woe', 'hash', 'label'].map(method => ({ method })),
  imputation_node: [{ method: 'simple', strategy: 'constant' }, { method: 'knn' }, { method: 'iterative' }],
  scale_numeric_features: ['standard', 'minmax', 'robust', 'maxabs'].map(method => ({ method })),
  outlier: ['iqr', 'zscore', 'winsorize', 'elliptic_envelope'].map(method => ({ method })),
  BinningNode: [{ strategy: 'custom', columns: ['age', 'income'] }, { label_format: 'range', drop_original: false }],
  feature_selection: ['variance_threshold', 'correlation_threshold', 'select_k_best', 'select_percentile',
    'select_fpr', 'select_fdr', 'select_fwe', 'generic_univariate_select', 'select_from_model', 'rfe'].map(method => ({ method })),
  ResamplingNode: [
    ...['smote', 'borderline_smote', 'svm_smote', 'kmeans_smote', 'adasyn', 'random_over_sampling'].map(method => ({ method, type: 'oversampling' })),
    ...['random_under_sampling', 'nearmiss', 'edited_nearest_neighbours'].map(method => ({ method, type: 'undersampling' })),
  ],
  TimeSeriesNode: ['lag', 'rolling', 'date'].map(method => ({ method })),
  casting: [{ column_types: { age: 'float', income: 'int' } }],
  TransformationNode: [{ transformations: ['yeo-johnson', 'exponential'].map(method => ({ method, columns: ['age'] })) }],
  FeatureGenerationNode: [{ operations: ['arithmetic', 'ratio', 'similarity', 'group_agg', 'datetime_extract'].map(operation_type => ({
    operation_type, method: operation_type === 'arithmetic' ? 'add' : 'mean', input_columns: ['age'], secondary_columns: ['income'],
    datetime_features: ['year'], isExpanded: true,
  })) }],
  TextCleaning: [{ operations: [
    { op: 'trim', mode: 'both' }, { op: 'case', mode: 'lower' }, { op: 'remove_special', mode: 'letters_only' },
    { op: 'regex', mode: 'custom', pattern: 'x', repl: 'y' },
  ] }],
  value_replacement: [{ replacements: [
    { old: 'a', new: 'b', oldType: 'string', newType: 'string' },
    { old: 1, new: 2, oldType: 'number', newType: 'number' },
    { old: true, new: false, oldType: 'boolean', newType: 'boolean' },
  ] }],
  AliasReplacement: [{ mode: 'custom', custom_pairs: { USA: 'US', GBR: 'GB' } }],
  InvalidValueReplacement: [{ mode: 'custom_range' }],
};

const cases = definitions.flatMap(definition => [undefined, ...(variants[definition.type] ?? [])].map((variant, index) => ({
  definition, name: `${definition.type} variant ${index}`, config: { ...definition.getDefaultConfig(), ...variant, isExpanded: true },
})));

describe('preprocessing settings accessible controls', () => {
  it.each(cases)('$name names every form control and action', ({ definition, config }) => {
    // Every rendered settings variant must expose a usable name without relying on a placeholder.
    const Settings = definition.settings;
    const { container } = render(<Settings config={config} onChange={() => {}} nodeId="settings-node" />);
    if (definition.type === 'TransformationNode') {
      for (const header of container.querySelectorAll('button[aria-expanded]')) fireEvent.click(header);
    }
    const controls = Array.from(container.querySelectorAll('input, select, textarea, button'));
    for (const control of controls) expect(control).toHaveAccessibleName();
    const unnamed = controls.filter(control => control.tagName !== 'BUTTON' && !control.getAttribute('aria-label') && !control.getAttribute('aria-labelledby') && !(control as HTMLInputElement).labels?.length);
    expect(unnamed.map(control => control.outerHTML.slice(0, 220))).toEqual([]);
  });

  it('keeps operation header controls independent during keyboard use', () => {
    // Pressing Space inside a method select must not collapse its operation or change pipeline data.
    const definition = definitions.find(node => node.type === 'FeatureGenerationNode')!;
    const Settings = definition.settings;
    const config = variants.FeatureGenerationNode![0]!;
    const onChange = vi.fn();
    render(<Settings config={config} onChange={onChange} />);
    const method = screen.getByRole('combobox', { name: 'Method for operation 1' });
    fireEvent.keyDown(method, { key: ' ' });
    expect(onChange).not.toHaveBeenCalled();
    const toggle = screen.getByRole('button', { name: 'Collapse arithmetic operation 1' });
    expect(toggle.tagName).toBe('BUTTON');
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Remove operation 2' }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ operations: (config.operations as unknown[]).filter((_, index) => index !== 1) });
  });

  it('distinguishes replacement fields and removes only the requested row', () => {
    // Row-qualified names must address the correct typed value and preserve neighboring replacements.
    const definition = definitions.find(node => node.type === 'value_replacement')!;
    const Settings = definition.settings;
    const config = { ...definition.getDefaultConfig(), ...variants.value_replacement![0] };
    const onChange = vi.fn();
    render(<Settings config={config} onChange={onChange} />);
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Find value for replacement 2' }), { target: { value: '7' } });
    expect(onChange).toHaveBeenLastCalledWith({ ...config, replacements: config.replacements.map((item: Record<string, unknown>, index: number) => index === 1 ? { ...item, old: 7 } : item) });
    fireEvent.click(screen.getByRole('button', { name: 'Remove replacement 3' }));
    expect(onChange).toHaveBeenLastCalledWith({ ...config, replacements: config.replacements.slice(0, 2) });
  });

  it('keeps resampling target suggestions local to their settings instance', async () => {
    // Two settings forms must not share a listbox ID or resolve suggestions from the other form.
    const definition = definitions.find(node => node.type === 'ResamplingNode')!;
    const Settings = definition.settings;
    const config = definition.getDefaultConfig();
    render(<><Settings config={config} onChange={() => {}} /><Settings config={config} onChange={() => {}} /></>);
    const controls = screen.getAllByRole('combobox', { name: 'Target Column' }) as HTMLInputElement[];
    fireEvent.focus(controls[0]!);
    const firstList = await screen.findByRole('listbox');
    expect(controls[0]).toHaveAttribute('aria-controls', firstList.id);
    expect(controls[1]).not.toHaveAttribute('aria-controls');
    fireEvent.keyDown(controls[0]!, { key: 'Escape' });
    fireEvent.focus(controls[1]!);
    const secondList = await screen.findByRole('listbox');
    expect(controls[1]).toHaveAttribute('aria-controls', secondList.id);
    expect(controls[0]).not.toHaveAttribute('aria-controls');
    expect(secondList.id).not.toBe(firstList.id);
  });

  it.each([{ method: 'rolling', name: 'mean' }, { method: 'date', name: 'year' }])('exposes selected $method options', ({ method, name }) => {
    // Keyboard users must be able to identify an option's current selection state before toggling it.
    const definition = definitions.find(node => node.type === 'TimeSeriesNode')!;
    const Settings = definition.settings;
    render(<Settings config={{ ...definition.getDefaultConfig(), method }} onChange={() => {}} />);
    expect(screen.getByRole('button', { name, pressed: true })).toBeInTheDocument();
  });

  it.each(['BinningNode', 'feature_selection', 'TransformationNode'])('keeps %s checkbox labels local to each settings instance', type => {
    // Repeated settings instances must never point a visible label at another instance's checkbox.
    const definition = definitions.find(node => node.type === type)!;
    const Settings = definition.settings;
    const config = { ...definition.getDefaultConfig(), ...variants[type]?.[0] };
    const { container } = render(<><section><Settings config={config} onChange={() => {}} /></section><section><Settings config={config} onChange={() => {}} /></section></>);
    if (type === 'TransformationNode') {
      for (const header of container.querySelectorAll('[role="button"], button[aria-expanded]')) fireEvent.click(header);
    }
    for (const section of container.querySelectorAll('section')) {
      const checkboxes = within(section).queryAllByRole('checkbox');
      expect(checkboxes.length).toBeGreaterThan(0);
      for (const checkbox of checkboxes) {
        const labels = Array.from((checkbox as HTMLInputElement).labels ?? []);
        expect(labels.length).toBeGreaterThan(0);
        expect(labels.every(label => section.contains(label))).toBe(true);
      }
    }
    const ids = Array.from(container.querySelectorAll('[id]')).map(control => control.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
