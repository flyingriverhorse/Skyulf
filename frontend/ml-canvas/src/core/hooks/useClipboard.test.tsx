import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import { useClipboard } from './useClipboard';
import { useGraphStore } from '../store/useGraphStore';
import { initializeRegistry } from '../registry/init';
import { useViewStore } from '../store/useViewStore';

// Mock React Flow's `useReactFlow` so its `getNodes()` / `getEdges()`
// read from our zustand store. The hook is decoupled from any actually
// rendered ReactFlow instance, which keeps the test focused on
// clipboard semantics rather than canvas wiring.
vi.mock('@xyflow/react', async () => {
  const actual = await vi.importActual<typeof import('@xyflow/react')>('@xyflow/react');
  return {
    ...actual,
    useReactFlow: () => ({
      getNodes: () => useGraphStore.getState().nodes,
      getEdges: () => useGraphStore.getState().edges,
    }),
  };
});

const Host: React.FC = () => {
  useClipboard();
  return <div data-testid="host" />;
};

const renderHost = () => render(<Host />);

describe('useClipboard', () => {
  beforeAll(() => initializeRegistry());
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
    useViewStore.setState({ readOnlyOverride: 'off' });
  });
  afterEach(() => {
    useViewStore.setState({ readOnlyOverride: 'auto' });
    vi.unstubAllGlobals();
  });

  it.each([
    { override: 'on', width: 1440, modifier: 'ctrlKey' },
    { override: 'on', width: 1440, modifier: 'metaKey' },
    { override: 'auto', width: 900, modifier: 'ctrlKey' },
  ] as const)('copies but blocks paste in $override mode at $width using $modifier', ({ override, width, modifier }) => {
    // Read-only copy must remain usable after editing resumes without advancing the paste offset.
    const id = useGraphStore.getState().addNode('imputation_node', { x: 100, y: 100 });
    useGraphStore.temporal.getState().clear();
    renderHost();
    vi.stubGlobal('innerWidth', width);
    useViewStore.setState({ readOnlyOverride: override });
    fireEvent.keyDown(document, { key: 'c', [modifier]: true });
    fireEvent.keyDown(document, { key: 'v', [modifier]: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
    useViewStore.setState({ readOnlyOverride: 'off' });
    fireEvent.keyDown(document, { key: 'v', [modifier]: true });
    expect(useGraphStore.getState().nodes).toHaveLength(2);
    expect(useGraphStore.getState().nodes.find(node => node.id !== id)?.position).toEqual({ x: 150, y: 150 });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
  });

  it('rechecks automatic read-only mode when the viewport shrinks after copy', () => {
    // Clipboard mutations must honor the latest viewport even without a hook rerender.
    useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    renderHost();
    vi.stubGlobal('innerWidth', 1440);
    useViewStore.setState({ readOnlyOverride: 'auto' });
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
    vi.stubGlobal('innerWidth', 900);
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });

  it('Ctrl+C with no selection is a no-op (paste also does nothing)', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    // Make sure it's not selected.
    useGraphStore.setState({
      nodes: useGraphStore.getState().nodes.map((n) => ({ ...n, selected: false })),
    });

    renderHost();
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });

    // Only the original node remains.
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    expect(useGraphStore.getState().nodes[0]?.id).toBe(id);
  });

  it('Ctrl+C then Ctrl+V duplicates selected nodes with a position offset', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 100, y: 100 });
    useGraphStore.setState({
      nodes: useGraphStore.getState().nodes.map((n) => ({ ...n, selected: n.id === id })),
    });

    renderHost();
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });

    const nodes = useGraphStore.getState().nodes;
    expect(nodes).toHaveLength(2);
    const clone = nodes.find((n) => n.id !== id)!;
    // First paste is offset by 50px.
    expect(clone.position).toEqual({ x: 150, y: 150 });
    // The clone gets a fresh deselected state.
    expect(clone.selected).toBe(false);
  });

  it('successive Ctrl+V pastes accumulate the offset', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    useGraphStore.setState({
      nodes: useGraphStore.getState().nodes.map((n) => ({ ...n, selected: n.id === id })),
    });

    renderHost();
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });

    const nodes = useGraphStore.getState().nodes;
    expect(nodes).toHaveLength(3);
    const positions = nodes.map((n) => n.position).sort((a, b) => a.x - b.x);
    // Original at (0,0); first paste at (50,50); second paste at (100,100).
    expect(positions).toEqual([
      { x: 0, y: 0 },
      { x: 50, y: 50 },
      { x: 100, y: 100 },
    ]);
  });

  it('does NOT trigger when focus is in an input', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    useGraphStore.setState({
      nodes: useGraphStore.getState().nodes.map((n) => ({ ...n, selected: n.id === id })),
    });

    renderHost();
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();

    fireEvent.keyDown(input, { key: 'c', ctrlKey: true, bubbles: true });
    fireEvent.keyDown(input, { key: 'v', ctrlKey: true, bubbles: true });

    expect(useGraphStore.getState().nodes).toHaveLength(1);
    input.remove();
  });

  it('copying multiple selected nodes also copies the internal edge between them', () => {
    const a = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    const b = useGraphStore.getState().addNode('encoding', { x: 100, y: 0 });
    useGraphStore.setState({
      nodes: useGraphStore.getState().nodes.map((n) => ({ ...n, selected: true })),
      edges: [{ id: 'e', source: a, target: b }],
    });

    renderHost();
    fireEvent.keyDown(document, { key: 'c', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'v', ctrlKey: true });

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(4);
    // 1 original edge + 1 cloned edge with remapped endpoints.
    expect(state.edges).toHaveLength(2);
    const cloned = state.edges.find((e) => e.id !== 'e')!;
    expect(cloned.source).not.toBe(a);
    expect(cloned.target).not.toBe(b);
  });
});
