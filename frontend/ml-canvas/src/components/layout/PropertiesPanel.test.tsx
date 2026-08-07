import { describe, expect, it, beforeEach, beforeAll } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PropertiesPanel } from './PropertiesPanel';
import { useGraphStore } from '../../core/store/useGraphStore';
import { initializeRegistry } from '../../core/registry/init';

const MERGE_NODE = {
  id: 'merge-node',
  type: 'custom',
  position: { x: 0, y: 0 },
  selected: true,
  data: { definitionType: 'MissingIndicator', label: 'MissingIndicator', columns: [] },
};

const EDGES = [
  { id: 'e1', source: 'branch-a', target: 'merge-node' },
  { id: 'e2', source: 'branch-b', target: 'merge-node' },
];

/** Seed the store with a two-branch fan-in and an optional engine advisory. */
const seedStore = (mergeWarnings: unknown[]) => {
  useGraphStore.setState({
    nodes: [MERGE_NODE],
    edges: EDGES,
    executionResult: mergeWarnings.length
      ? ({ merge_warnings: mergeWarnings, node_results: {} } as never)
      : null,
  } as never);
};

describe('PropertiesPanel merge strategy', () => {
  beforeAll(() => initializeRegistry());

  const renderPanel = () =>
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PropertiesPanel />
      </QueryClientProvider>,
    );

  beforeEach(() => {
    useGraphStore.setState({ executionResult: null } as never);
  });

  it('hides the strategy control when no branch contested a column', () => {
    seedStore([]);
    renderPanel();
    expect(screen.queryByText('Merge Strategy')).not.toBeInTheDocument();
  });

  it('predicts a conflict before any run when both branches target the same column', () => {
    useGraphStore.setState({
      nodes: [
        MERGE_NODE,
        { id: 'branch-a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'TransformationNode', label: 'TransformationNode', transformations: [{ columns: ['SepalLengthCm'], method: 'log' }] } },
        { id: 'branch-b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'TransformationNode', label: 'TransformationNode2', transformations: [{ columns: ['SepalLengthCm'], method: 'cube' }] } },
      ],
      edges: EDGES,
      executionResult: null,
    } as never);

    renderPanel();
    expect(screen.getByText('Merge Strategy')).toBeTruthy();
    expect(screen.getByText('Predicted')).toBeTruthy();
    expect(screen.getByText(/SepalLengthCm/)).toBeTruthy();
  });

  it('hides it even when a run produced only an upstream-drop advisory', () => {
    seedStore([
      { node_id: 'merge-node', kind: 'upstream_drop_reapplied', dropped_columns: ['Id'] },
    ]);
    renderPanel();
    expect(screen.queryByText('Merge Strategy')).not.toBeInTheDocument();
  });

  it('shows it, naming the contested columns, once two branches edited the same column', () => {
    useGraphStore.setState({
      nodes: [
        MERGE_NODE,
        { id: 'branch-a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'drop_columns', label: 'Drop Missing Columns' } },
        { id: 'branch-b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'SimpleTransformation', label: 'TransformationNode' } },
      ],
      edges: EDGES,
      executionResult: {
        node_results: {},
        merge_warnings: [
          {
            node_id: 'merge-node',
            kind: 'sibling_fan_in',
            inputs: ['branch-a', 'branch-b'],
            overlap_columns: ['SepalLengthCm'],
            winner_input: 'branch-b',
          },
        ],
      },
    } as never);

    renderPanel();
    expect(screen.getByText('Merge Strategy')).toBeTruthy();
    expect(screen.getByText(/SepalLengthCm/)).toBeTruthy();
    expect(screen.getByText('Drop Missing Columns')).toBeTruthy();
    expect(screen.getByText('TransformationNode')).toBeTruthy();
    expect(screen.getByText(/Keep TransformationNode \(last connected/)).toBeTruthy();
    expect(screen.getByText(/Keep Drop Missing Columns \(first connected/)).toBeTruthy();
  });
});
