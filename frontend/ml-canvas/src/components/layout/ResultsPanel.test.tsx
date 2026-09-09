import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ResultsPanel } from './ResultsPanel';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { initializeRegistry } from '../../core/registry/init';

vi.mock('../shared', async () => {
  const actual = await vi.importActual<typeof import('../shared')>('../shared');
  return {
    ...actual,
    useConfirm: () => vi.fn(),
  };
});

describe('ResultsPanel', () => {
  beforeEach(() => {
    initializeRegistry();
    useGraphStore.setState({
      nodes: [],
      edges: [],
      executionResult: null,
      lastRunError: null,
    });
    useViewStore.setState({
      isResultsPanelExpanded: true,
      isResultsPanelDismissed: false,
    });
  });

  it('shows validation issues and lets the user select the offending node', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'orphan-encoding',
          type: 'custom',
          position: { x: 100, y: 0 },
          data: { definitionType: 'encoding', method: 'label', columns: ['status'] },
        },
      ],
      [],
    );

    render(<ResultsPanel />);

    const issueButton = screen.getByRole('button', { name: /encoding/i });
    fireEvent.click(issueButton);

    expect(useGraphStore.getState().nodes.find((node) => node.id === 'orphan-encoding')?.selected).toBe(true);
  });

  it('keeps the last preview error visible in the results panel', () => {
    useGraphStore.setState({ lastRunError: 'Backend exploded' });

    render(<ResultsPanel />);

    expect(screen.getByRole('alert')).toHaveTextContent('Backend exploded');
  });

  it('never reports a row count when no preview run has produced results', () => {
    useGraphStore.setState({ lastRunError: 'Backend exploded' });

    render(<ResultsPanel />);

    expect(screen.queryByText(/rows/i)).not.toBeInTheDocument();
  });

  it('announces only the issue count so editing the graph does not re-read every issue', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'orphan-encoding',
          type: 'custom',
          position: { x: 100, y: 0 },
          data: { definitionType: 'encoding', method: 'label', columns: ['status'] },
        },
      ],
      [],
    );

    render(<ResultsPanel />);

    const issueButton = screen.getByRole('button', { name: /encoding/i });
    const liveRegion = issueButton.closest('[role="alert"], [role="status"], [aria-live]');
    expect(liveRegion).toBeNull();
  });

  it('closes preview results with the X button, clearing the run data', () => {
    useGraphStore.setState({
      executionResult: {
        pipeline_id: 'p1',
        status: 'success',
        node_results: {},
        preview_data: { Result: [{ a: 1 }] },
        recommendations: [],
      },
      lastRunError: 'stale error',
    });

    render(<ResultsPanel />);

    fireEvent.click(screen.getByRole('button', { name: 'Close preview results' }));

    expect(useGraphStore.getState().executionResult).toBeNull();
    expect(useGraphStore.getState().lastRunError).toBeNull();
  });

  it('closes a validation-only panel with the X button until the issues change', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'orphan-encoding',
          type: 'custom',
          position: { x: 100, y: 0 },
          data: { definitionType: 'encoding', method: 'label', columns: ['status'] },
        },
      ],
      [],
    );

    const { container } = render(<ResultsPanel />);
    expect(screen.getByText('Preview Results')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Close preview results' }));

    expect(container.querySelector('.absolute.bottom-0')).toBeNull();
  });

  /** Canvas-only safety feedback must not create an empty results panel or hide existing data. */
  it.each([false, true])('omits leakage issues while preserving existing data: %s', withResults => {
    useGraphStore.setState({
      nodes: [
        { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
        { id: 'imputer', position: { x: 100, y: 0 }, data: { definitionType: 'imputation_node', columns: ['value'], strategy: 'mean' } },
        { id: 'split', position: { x: 200, y: 0 }, data: { definitionType: 'TrainTestSplitter', test_size: 0.2, validation_size: 0, random_state: 42, stratify: false, shuffle: true } },
      ],
      edges: [{ id: 'e1', source: 'dataset', target: 'imputer' }, { id: 'e2', source: 'imputer', target: 'split' }],
      executionResult: withResults ? { pipeline_id: 'earlier', status: 'success', node_results: {}, preview_data: [{ value: 3 }], recommendations: [] } : null,
    });
    render(<ResultsPanel />);
    expect(screen.queryByText('Validation issues')).not.toBeInTheDocument();
    if (withResults) {
      expect(screen.getByRole('region', { name: 'Preview results' })).toBeVisible();
      expect(screen.getByRole('tab', { name: 'Data' })).toHaveAttribute('aria-selected', 'true');
      expect(screen.getByText('3')).toBeInTheDocument();
    } else {
      expect(screen.queryByRole('region', { name: 'Preview results' })).not.toBeInTheDocument();
    }
  });

  /** Mixed graphs must retain actionable ordinary issues without duplicating canvas leakage feedback. */
  it('keeps configuration errors in the panel when leakage also exists', () => {
    useGraphStore.setState({
      nodes: [
        { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
        { id: 'imputer', position: { x: 100, y: 0 }, data: { definitionType: 'imputation_node', columns: ['value'], strategy: 'mean' } },
        { id: 'split', position: { x: 200, y: 0 }, data: { definitionType: 'TrainTestSplitter', test_size: 0.2, validation_size: 0, random_state: 42, stratify: false, shuffle: true } },
        { id: 'unknown', position: { x: 300, y: 0 }, data: { definitionType: 'unknown_node' } },
      ],
      edges: [{ id: 'e1', source: 'dataset', target: 'imputer' }, { id: 'e2', source: 'imputer', target: 'split' }],
    });
    render(<ResultsPanel />);
    expect(screen.getByText('Validation issues')).toBeVisible();
    expect(screen.queryByText('leakage')).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('1 validation issue blocking preview');
  });

  /** Leakage-only edits must respect dismissal even if prior preview data is still available. */
  it('does not reopen a dismissed data panel when only leakage issues change', () => {
    useGraphStore.setState({
      nodes: [
        { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
        { id: 'drop', position: { x: 100, y: 100 }, data: { definitionType: 'drop_missing_columns', columns: ['id'], missing_threshold: 0 } },
      ],
      edges: [{ id: 'e0', source: 'dataset', target: 'drop' }],
      executionResult: { pipeline_id: 'earlier', status: 'success', node_results: {}, preview_data: [{ value: 3 }], recommendations: [] },
    });
    render(<ResultsPanel />);
    act(() => useViewStore.getState().setResultsPanelDismissed(true));
    act(() => useGraphStore.setState(state => ({
      nodes: [...state.nodes,
        { id: 'imputer', position: { x: 100, y: 0 }, data: { definitionType: 'imputation_node', columns: ['value'], strategy: 'mean' } },
        { id: 'split', position: { x: 200, y: 0 }, data: { definitionType: 'TrainTestSplitter', test_size: 0.2, validation_size: 0, random_state: 42, stratify: false, shuffle: true } },
      ],
      edges: [...state.edges, { id: 'e1', source: 'dataset', target: 'imputer' }, { id: 'e2', source: 'imputer', target: 'split' }],
    })));
    expect(useGraphStore.getState().executionResult?.pipeline_id).toBe('earlier');
    expect(useViewStore.getState().isResultsPanelDismissed).toBe(true);
    expect(screen.queryByRole('region', { name: 'Preview results' })).not.toBeInTheDocument();
  });
});
