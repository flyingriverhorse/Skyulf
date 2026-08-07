import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import type { NodeSettingsProps } from '../../../core/types/nodes';
import { initializeRegistry } from '../../../core/registry/init';
import { FeatureGenerationNode } from './FeatureGenerationNode';

type FeatureGenerationConfig = ReturnType<typeof FeatureGenerationNode.getDefaultConfig>;

const mockGraphState = {
  nodes: [],
  edges: [],
  executionResult: {
    node_results: {},
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
      columns: {
        age: { name: 'age', dtype: 'int' },
        income: { name: 'income', dtype: 'float' },
        signup_date: { name: 'signup_date', dtype: 'date' },
      },
    },
    isLoading: false,
  }),
}));

vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => new Set<string>(),
}));

vi.mock('../../../core/hooks/useRecommendations', () => ({
  useRecommendations: () => recommendationFixture,
}));

vi.mock('../../../core/store/useGraphStore', () => ({
  useGraphStore: (selector: (state: typeof mockGraphState) => unknown) => selector(mockGraphState),
}));

vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), false] as const,
}));

vi.mock('../shared/ColumnMultiSelect', () => ({
  ColumnMultiSelect: ({
    label,
    selected,
  }: {
    label: string;
    selected: string[];
  }) => (
    <div data-testid={`column-select-${label}`}>
      {label}: {selected.join(',')}
    </div>
  ),
}));

function renderFeatureGeneration(config: FeatureGenerationConfig, onChange = vi.fn()) {
  const client = new QueryClient();
  const Settings = FeatureGenerationNode.settings as React.JSXElementConstructor<NodeSettingsProps<FeatureGenerationConfig>>;

  render(
    <QueryClientProvider client={client}>
      <Settings config={config} onChange={onChange} nodeId="feature-node" />
    </QueryClientProvider>,
  );

  return { onChange };
}

describe('FeatureGenerationNode recommendations', () => {
  beforeAll(() => initializeRegistry());

  beforeEach(() => {
    mockGraphState.nodes = [];
    mockGraphState.edges = [];
    mockGraphState.executionResult = { node_results: {} };
  });

  it('keeps feature-generation recommendations informational and hides the Apply action', () => {
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
