import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
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
});
