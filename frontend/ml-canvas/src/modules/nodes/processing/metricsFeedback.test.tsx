import React from 'react';
import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { NodeSettingsProps } from '../../../core/types/nodes';
import { FeatureSelectionNode } from './FeatureSelectionNode';
import { OutlierNode } from './OutlierNode';
import { ScalingNode } from './ScalingNode';

const mockGraphState = {
  nodes: [],
  edges: [],
  executionResult: {
    node_results: {} as Record<string, { status?: string; metrics?: Record<string, unknown> }>,
  },
};

vi.mock('../../../core/hooks/useUpstreamData', () => ({
  useUpstreamData: () => [],
}));

vi.mock('../../../core/hooks/useDatasetSchema', () => ({
  useDatasetSchema: () => ({ data: null, isLoading: false }),
}));

vi.mock('../../../core/store/useGraphStore', () => ({
  useGraphStore: (selector: (state: typeof mockGraphState) => unknown) => selector(mockGraphState),
}));

vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({
  useUpstreamDroppedColumns: () => new Set<string>(),
}));

vi.mock('../../../core/hooks/useRecommendations', () => ({
  useRecommendations: () => [],
}));

vi.mock('../../../components/panels/RecommendationsPanel', () => ({
  RecommendationsPanel: () => <div data-testid="recommendations-panel" />,
}));

vi.mock('../shared/ColumnMultiSelect', () => ({
  ColumnMultiSelect: () => <div data-testid="column-multi-select" />,
}));

vi.mock('../../../core/hooks/useIsWideContainer', () => ({
  useIsWideContainer: () => [React.createRef<HTMLDivElement>(), false] as const,
}));

vi.mock('@xyflow/react', async () => {
  const actual = await vi.importActual<typeof import('@xyflow/react')>('@xyflow/react');
  return {
    ...actual,
    getIncomers: () => [],
  };
});

type ScalingConfig = ReturnType<typeof ScalingNode.getDefaultConfig>;
type OutlierConfig = ReturnType<typeof OutlierNode.getDefaultConfig>;
type FeatureSelectionConfig = ReturnType<typeof FeatureSelectionNode.getDefaultConfig>;

function renderScaling(metrics: Record<string, unknown>, config: ScalingConfig) {
  mockGraphState.executionResult.node_results = {
    'node-1': { metrics },
  };
  const Settings = ScalingNode.settings as React.JSXElementConstructor<NodeSettingsProps<ScalingConfig>>;
  render(<Settings config={config} onChange={() => {}} nodeId="node-1" />);
}

function renderOutlier(metrics: Record<string, unknown>, config: OutlierConfig) {
  mockGraphState.executionResult.node_results = {
    'node-1': { metrics },
  };
  const Settings = OutlierNode.settings as React.JSXElementConstructor<NodeSettingsProps<OutlierConfig>>;
  render(<Settings config={config} onChange={() => {}} nodeId="node-1" />);
}

function renderFeatureSelection(metrics: Record<string, unknown>, config: FeatureSelectionConfig) {
  mockGraphState.executionResult.node_results = {
    'node-1': { status: 'success', metrics },
  };
  const Settings = FeatureSelectionNode.settings as React.JSXElementConstructor<NodeSettingsProps<FeatureSelectionConfig>>;
  render(<Settings config={config} onChange={() => {}} nodeId="node-1" />);
}

describe('preprocessing node feedback metrics', () => {
  beforeEach(() => {
    mockGraphState.nodes = [];
    mockGraphState.edges = [];
    mockGraphState.executionResult.node_results = {};
  });

  it('renders ScalingNode feedback from wrapped single-step metrics', () => {
    renderScaling(
      {
        fit_time: 0.3,
        steps: {
          '0:step': {
            name: 'step',
            transformer: 'StandardScaler',
            details: {
              columns: ['age'],
              mean: [0],
              scale: [1],
            },
          },
        },
      },
      {
        columns: ['age'],
        method: 'standard',
      },
    );

    expect(screen.getByText('Scaling Statistics')).toBeInTheDocument();
    expect(screen.getByText('age')).toBeInTheDocument();
    expect(screen.getByText('μ=0.00, σ=1.00')).toBeInTheDocument();
  });

  it('keeps ScalingNode legacy flat metrics rendering', () => {
    renderScaling(
      {
        columns: ['fare'],
        data_min: [1],
        data_max: [10],
      },
      {
        columns: ['fare'],
        method: 'minmax',
      },
    );

    expect(screen.getByText('fare')).toBeInTheDocument();
    expect(screen.getByText('Min=1.00, Max=10.00')).toBeInTheDocument();
  });

  it('renders OutlierNode feedback from wrapped single-step metrics', () => {
    renderOutlier(
      {
        fit_time: 0.2,
        steps: {
          '0:step': {
            name: 'step',
            transformer: 'IQR',
            details: {
              rows_removed: 2,
              rows_remaining: 8,
              rows_total: 10,
              bounds: {
                age: { lower: 1, upper: 9 },
              },
            },
          },
        },
      },
      {
        columns: ['age'],
        method: 'iqr',
      },
    );

    expect(screen.getByText('Execution Feedback')).toBeInTheDocument();
    expect(screen.getByText('Rows Removed')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('[1.00, 9.00]')).toBeInTheDocument();
  });

  it('keeps OutlierNode legacy flat metrics rendering', () => {
    renderOutlier(
      {
        values_clipped: 3,
        warnings: ['Used percentile bounds'],
      },
      {
        columns: ['age'],
        method: 'winsorize',
      },
    );

    expect(screen.getByText('Values Clipped')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('Used percentile bounds')).toBeInTheDocument();
  });

  it('does not claim zero dropped columns for ambiguous wrapped FeatureSelection metrics', () => {
    renderFeatureSelection(
      {
        steps: {
          '0:before': {
            details: { dropped_columns: ['age'] },
          },
          '1:after': {
            details: { dropped_columns: ['fare'] },
          },
        },
      },
      {
        method: 'variance_threshold',
        threshold: 0.1,
        k: 10,
      },
    );

    expect(screen.getByText(/could not be resolved to a single step/i)).toBeInTheDocument();
    expect(screen.queryByText('No columns were dropped.')).not.toBeInTheDocument();
  });

  it('keeps FeatureSelection empty wrapped single-step feedback as zero drops', () => {
    renderFeatureSelection(
      {
        steps: {
          '0:feature_selection': {
            details: {},
          },
        },
      },
      {
        method: 'variance_threshold',
        threshold: 0.1,
        k: 10,
      },
    );

    expect(screen.getByText('No columns were dropped.')).toBeInTheDocument();
  });

  it('keeps FeatureSelection legacy flat dropped-columns feedback', () => {
    renderFeatureSelection(
      {
        dropped_columns: ['fare'],
      },
      {
        method: 'variance_threshold',
        threshold: 0.1,
        k: 10,
      },
    );

    expect(screen.getByText('Dropped Columns (1)')).toBeInTheDocument();
    expect(screen.getByText('fare')).toBeInTheDocument();
  });
});
