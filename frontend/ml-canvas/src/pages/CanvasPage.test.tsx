import { render, screen, fireEvent } from '@testing-library/react';
import { StrictMode } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { describe, expect, it, vi, beforeAll, beforeEach } from 'vitest';
import type { Node } from '@xyflow/react';

import { CanvasPage } from './CanvasPage';
import { useGraphStore } from '../core/store/useGraphStore';
import { FOCUS_NODE_EVENT } from '../core/hooks/useKeyboardShortcuts';
import { initializeRegistry } from '../core/registry/init';

// MainLayout mounts the full canvas app (React Flow, Sidebar, Toolbar,
// autosave, ...) which is unrelated to the deep-link selection/notice
// logic under test here and heavy to render under jsdom — stub it so
// these tests exercise only CanvasPage's own effects and markup.
vi.mock('../components/layout/MainLayout', () => ({
  MainLayout: () => <div data-testid="main-layout" />,
}));

const NODE_A: Node = {
  id: 'encoding-26e140ef-766f-438e-860b-8199c080fc92',
  type: 'custom',
  position: { x: 0, y: 0 },
  data: { definitionType: 'encoding' },
};

function renderCanvas(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <CanvasPage />
    </MemoryRouter>,
  );
}

/** Expose the router's resulting URL without mocking its navigation behavior. */
function CurrentSearch() {
  return <output aria-label="Current query">{useLocation().search}</output>;
}

beforeEach(() => {
  useGraphStore.setState({ nodes: [], edges: [] } as never);
});

describe('CanvasPage dataset source links', () => {
  beforeAll(() => initializeRegistry());

  it('reuses an existing registered dataset and removes source_id in StrictMode', () => {
    // React Flow custom wrappers must not hide an existing source from duplicate detection.
    const id = useGraphStore.getState().addNode('dataset_node', { x: 30, y: 40 }, { datasetId: 'source-1' });
    useGraphStore.temporal.getState().clear();
    render(<StrictMode><MemoryRouter initialEntries={['/canvas?view=canvas&source_id=source-1&keep=yes']}>
      <CanvasPage /><CurrentSearch />
    </MemoryRouter></StrictMode>);
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.getState().nodes[0]).toMatchObject({ id, position: { x: 30, y: 40 } });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
    expect(screen.getByLabelText('Current query')).toHaveTextContent('?view=canvas&keep=yes');
  });

  it.each([
    { definitionType: 'dataset_node', datasetId: 'different-source' },
    { definitionType: 'imputation_node', datasetId: 'source-1' },
  ])('inserts the requested source alongside $definitionType using $datasetId', ({ definitionType, datasetId }) => {
    // Matching needs both the registered dataset definition and the requested source identity.
    useGraphStore.getState().addNode(definitionType, { x: 30, y: 40 }, { datasetId });
    render(<StrictMode><MemoryRouter initialEntries={['/canvas?view=canvas&source_id=source-1']}>
      <CanvasPage /><CurrentSearch />
    </MemoryRouter></StrictMode>);
    expect(useGraphStore.getState().nodes).toHaveLength(2);
    expect(useGraphStore.getState().nodes[1]?.data).toMatchObject({ definitionType: 'dataset_node', datasetId: 'source-1' });
    expect(screen.getByLabelText('Current query')).toHaveTextContent('?view=canvas');
  });
});

describe('CanvasPage node RecordLink handling (OPS-007)', () => {
  it('selects and focuses a node that exists in the currently loaded graph', () => {
    useGraphStore.setState({ nodes: [NODE_A], edges: [] } as never);
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent');

    renderCanvas(`/canvas?oc.kind=node&oc.nodeId=${NODE_A.id}&oc.pipelineId=pipe-1`);

    const selected = useGraphStore.getState().nodes.find((n) => n.id === NODE_A.id);
    expect(selected?.selected).toBe(true);

    const focusEvent = dispatchSpy.mock.calls
      .map((args) => args[0])
      .find((event): event is CustomEvent => event instanceof CustomEvent && event.type === FOCUS_NODE_EVENT);
    expect(focusEvent?.detail).toEqual({ id: NODE_A.id, focusWrapper: true });

    // No error/notice banner when the node was found.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('explains — rather than silently doing nothing — when the node belongs to a different pipeline', () => {
    // Graph loaded on canvas has no nodes at all, so the linked node
    // (which does carry a pipelineId) cannot be the current pipeline.
    renderCanvas('/canvas?oc.kind=node&oc.nodeId=node-99&oc.pipelineId=pipeline-other&oc.origin=%2Fslow-nodes');

    const notice = screen.getByRole('alert');
    expect(notice).toHaveTextContent('node-99');
    expect(notice).toHaveTextContent('pipeline-other');
    expect(notice).toHaveTextContent(/different|isn't the pipeline currently open/i);

    const backLink = screen.getByRole('link', { name: /back to slow nodes/i });
    expect(backLink).toHaveAttribute('href', '/slow-nodes');
  });

  it('names the missing node id when it cannot be found and no pipeline context is available', () => {
    renderCanvas('/canvas?oc.kind=node&oc.nodeId=ghost-node-1');

    const notice = screen.getByRole('alert');
    expect(notice).toHaveTextContent('ghost-node-1');
    expect(notice).toHaveTextContent(/could not be found/i);
  });

  it('is dismissible and non-blocking', () => {
    renderCanvas('/canvas?oc.kind=node&oc.nodeId=ghost-node-2');

    expect(screen.getByRole('alert')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    // The canvas underneath stays mounted/interactive throughout.
    expect(screen.getByTestId('main-layout')).toBeInTheDocument();
  });

  it('round-trips the origin so the user can return to where they came from', () => {
    renderCanvas('/canvas?oc.kind=node&oc.nodeId=ghost-node-3&oc.origin=%2Ferrors');

    const backLink = screen.getByRole('link', { name: /back to error log/i });
    expect(backLink).toHaveAttribute('href', '/errors');
  });
});
