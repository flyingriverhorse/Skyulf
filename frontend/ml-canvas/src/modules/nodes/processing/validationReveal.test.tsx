import React from 'react';
import { act, render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { registry } from '../../../core/registry/NodeRegistry';
import type { NodeDefinition } from '../../../core/types/nodes';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useViewStore } from '../../../core/store/useViewStore';
import { ValidationNavigation } from '../../../components/shared/ValidationField';
import { BinningNode } from './BinningNode';
import { FeatureGenerationNode } from './FeatureGenerationNode';
import { FeatureInteractionNode } from './FeatureInteractionNode';
import { PolynomialFeaturesNode } from './PolynomialFeaturesNode';
import { TransformationNode } from './TransformationNode';

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: () => ({ data: { columns: { age: { name: 'age', dtype: 'int' } } }, isLoading: false }),
}));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set() }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => [] }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), false],
}));
vi.mock('../shared/ColumnMultiSelect', () => ({
  ColumnMultiSelect: () => <select aria-label="Columns"><option value="">Select columns</option></select>,
}));

describe('processing validation reveals', () => {
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
    useViewStore.setState({ validationFocusRequest: null });
  });

  const cases: { definition: NodeDefinition; config: Record<string, unknown>; field: string }[] = [
    {
      definition: TransformationNode,
      config: { transformations: [{ method: 'log', columns: ['age'] }, { method: 'square', columns: [] }] },
      field: 'transformations.1.columns',
    },
    {
      definition: FeatureGenerationNode,
      config: { operations: [{ operation_type: 'arithmetic', method: 'add', input_columns: ['age'], secondary_columns: [], isExpanded: false }] },
      field: 'operations.0.secondary_columns',
    },
    {
      definition: PolynomialFeaturesNode,
      config: { ...PolynomialFeaturesNode.getDefaultConfig(), columns: ['age'], degree: 1, isExpanded: false },
      field: 'degree',
    },
    {
      definition: FeatureInteractionNode,
      config: { ...FeatureInteractionNode.getDefaultConfig(), isExpanded: false },
      field: 'columns',
    },
    {
      definition: BinningNode,
      config: { ...BinningNode.getDefaultConfig(), columns: ['age'], strategy: 'custom', n_bins: 1 },
      field: 'n_bins',
    },
  ];

  it.each(cases)('reveals $definition.type $field without changing config', async ({ definition, config, field }) => {
    // Issue navigation must open the actual hidden setting without changing pipeline data.
    registry.register(definition);
    useGraphStore.setState({
      nodes: [{ id: 'invalid', position: { x: 0, y: 0 }, data: { ...config, definitionType: definition.type } }],
    });
    const onChange = vi.fn();
    const Settings = definition.settings!;
    const { container } = render(
      <ValidationNavigation nodeId="invalid"><Settings config={config} onChange={onChange} nodeId="invalid" /></ValidationNavigation>,
    );
    expect(container.querySelector(`[data-validation-field="${field}"]`)).toBeNull();

    act(() => useViewStore.getState().requestValidationFocus({
      nodeId: 'invalid', nodeLabel: definition.label, category: 'configuration', field, message: 'Invalid configuration',
    }));

    await waitFor(() => {
      const control = container.querySelector(`[data-validation-field="${field}"]`)?.querySelector('input, select');
      expect(control).toHaveFocus();
      expect(control).toHaveAttribute('aria-invalid', 'true');
    });
    expect(onChange).not.toHaveBeenCalled();
  });
});
