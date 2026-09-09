import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ResultsPanel } from './ResultsPanel';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { initializeRegistry } from '../../core/registry/init';
import { FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';

const confirm = vi.hoisted(() => vi.fn());

vi.mock('../shared', async () => {
  const actual = await vi.importActual<typeof import('../shared')>('../shared');
  return {
    ...actual,
    useConfirm: () => confirm,
  };
});

describe('ResultsPanel', () => {
  beforeEach(() => {
    confirm.mockReset();
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
      isResultsPanelMaximized: false,
      resultsPanelHeight: 384,
      readOnlyOverride: 'off',
      validationFocusRequest: null,
    });
  });

  /** Branch-filtered advisories retain their expanded state and rewire only after confirmation. */
  it.each([false, true])('preserves branch advisories and respects confirmation %s', async accepted => {
    useGraphStore.setState({ executionResult: {
      pipeline_id: 'advisories', status: 'success', node_results: {}, recommendations: [], preview_data: null,
      branch_previews: { Left: [{ value: 1 }], Right: [{ value: 2 }] },
      branch_node_ids: { Left: ['left-consumer'], Right: ['right-consumer'] },
      merge_warnings: [
        { node_id: 'left-consumer', kind: 'sibling_fanin', inputs: ['a', 'b'], overlap_columns: ['value'], message: 'left warning' },
        { node_id: 'right-consumer', kind: 'sibling_fanin', inputs: ['c', 'd'], overlap_columns: ['value'], message: 'right warning' },
      ],
    } });
    const originalChain = useGraphStore.getState().chainSiblings;
    const chain = vi.fn().mockReturnValue(true);
    useGraphStore.setState({ chainSiblings: chain });
    confirm.mockResolvedValue(accepted);
    try {
      render(<ResultsPanel />);
      fireEvent.click(screen.getByRole('tab', { name: 'Steps 1' }));
      expect(screen.getByText('left-consumer')).toBeVisible();
      expect(screen.queryByText('right-consumer')).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('tab', { name: 'Issues 1' }));
      fireEvent.click(screen.getByRole('button', { name: /1 merge advisory/ }));
      fireEvent.click(screen.getByRole('tab', { name: 'Data' }));
      fireEvent.click(screen.getByRole('button', { name: /Right/ }));
      fireEvent.click(screen.getByRole('tab', { name: 'Issues 1' }));
      expect(screen.getByRole('button', { name: /1 merge advisory/ })).toHaveAttribute('aria-expanded', 'true');
      expect(screen.queryByText('left-consumer')).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Chain instead' }));
      await waitFor(() => expect(confirm).toHaveBeenCalledWith(expect.objectContaining({ title: 'Rewire as a linear chain?', confirmLabel: 'Rewire' })));
      if (accepted) expect(chain).toHaveBeenCalledWith('right-consumer', ['c', 'd']);
      else expect(chain).not.toHaveBeenCalled();
    } finally {
      useGraphStore.setState({ chainSiblings: originalChain });
    }
  });

  /** Branch changes must preserve shared split selection and prefer train, then X, for new splits. */
  it('retains branch order, split intent and true row totals across branches', () => {
    useGraphStore.setState({ executionResult: {
      pipeline_id: 'branches', status: 'success', node_results: {}, recommendations: [],
      preview_data: null,
      branch_previews: {
        Zebra: { test: [{ value: 'z-test' }], train: [{ value: 'z-train' }] },
        Alpha: { test: [{ value: 'a-test' }], train: [{ value: 'a-train' }] },
        Other: { y: [{ value: 'o-y' }], X: [{ value: 'o-x' }] },
      },
      branch_preview_totals: { Zebra: { train: 120 }, Alpha: { test: 40 }, Other: { _total: 75 } },
    } });
    render(<ResultsPanel />);
    expect(screen.getByText('z-train')).toBeVisible();
    expect(screen.getByText('1 of 120 rows shown · 3 branches')).toBeVisible();
    expect(screen.getAllByRole('button').filter(button => ['Zebra', 'Alpha', 'Other'].includes(button.textContent ?? '')).map(button => button.textContent)).toEqual(['Zebra', 'Alpha', 'Other']);
    fireEvent.click(screen.getByRole('button', { name: 'test 1' }));
    fireEvent.click(screen.getByRole('button', { name: 'Alpha' }));
    expect(screen.getByText('a-test')).toBeVisible();
    expect(screen.getByText('1 of 40 rows shown · 3 branches')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Other' }));
    expect(screen.getByText('o-x')).toBeVisible();
    expect(screen.getByText('1 of 75 rows shown · 3 branches')).toBeVisible();
  });

  /** Choosing a pane is durable UI intent across collapse, dismissal and fresh results. */
  it('retains the chosen pane when hidden and reopened by a new result', () => {
    const result = { pipeline_id: 'first', status: 'success', node_results: {}, preview_data: [{ value: 1 }], recommendations: [] };
    useGraphStore.setState({ executionResult: result });
    render(<ResultsPanel />);
    fireEvent.click(screen.getByRole('tab', { name: 'Steps' }));
    fireEvent.click(screen.getByRole('button', { name: 'Collapse results panel' }));
    fireEvent.click(screen.getByRole('button', { name: 'Expand results panel' }));
    expect(screen.getByRole('tab', { name: 'Steps' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Close preview results' }));
    expect(screen.queryByRole('region', { name: 'Preview results' })).not.toBeInTheDocument();
    act(() => useGraphStore.setState({ executionResult: { ...result, pipeline_id: 'second' } }));
    expect(screen.getByRole('tab', { name: 'Steps' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText('No steps ran. Run a preview to see which nodes executed.')).toBeVisible();
  });

  /** Read-only issue navigation selects the node without requesting editable settings focus. */
  it.each(['off', 'on'] as const)('routes validation focus with read-only override %s', override => {
    useGraphStore.setState({ nodes: [{ id: 'unknown', position: { x: 0, y: 0 }, data: { definitionType: 'unknown_node' } }] });
    useViewStore.setState({ readOnlyOverride: override });
    const focus = vi.fn();
    window.addEventListener(FOCUS_NODE_EVENT, focus);
    render(<ResultsPanel />);
    fireEvent.click(screen.getByRole('button', { name: /Unknown Node/i }));
    window.removeEventListener(FOCUS_NODE_EVENT, focus);
    expect(focus).toHaveBeenCalledWith(expect.objectContaining({ detail: { id: 'unknown', focusWrapper: override === 'on' } }));
    expect(useViewStore.getState().validationFocusRequest?.nodeId).toBe(override === 'on' ? undefined : 'unknown');
    expect(useGraphStore.getState().nodes[0]?.selected).toBe(true);
  });

  /** Home retains the preferred height even in a short viewport; other resize keys obey its cap. */
  it('preserves keyboard resize bounds and the Home height preference', () => {
    useGraphStore.setState({ lastRunError: 'Resize this panel' });
    render(<ResultsPanel maxHeight={300} />);
    const separator = screen.getByRole('separator', { name: 'Resize results panel' });
    fireEvent.keyDown(separator, { key: 'Home' });
    expect(useViewStore.getState().resultsPanelHeight).toBe(384);
    expect(separator).toHaveAttribute('aria-valuenow', '300');
    fireEvent.keyDown(separator, { key: 'ArrowDown' });
    expect(separator).toHaveAttribute('aria-valuenow', '280');
    fireEvent.keyDown(separator, { key: 'End' });
    expect(separator).toHaveAttribute('aria-valuenow', '300');
    fireEvent.click(screen.getByRole('button', { name: 'Maximize results panel' }));
    expect(screen.queryByRole('separator')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Restore results panel' }));
    expect(screen.getByRole('separator')).toHaveAttribute('aria-valuenow', '300');
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
