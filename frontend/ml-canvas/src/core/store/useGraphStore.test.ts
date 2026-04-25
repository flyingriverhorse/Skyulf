import { describe, it, expect, beforeAll, beforeEach, vi } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { useGraphStore } from './useGraphStore';
import { initializeRegistry } from '../registry/init';

// Pure-store smoke tests. These exercise the actions that don't depend on
// the NodeRegistry (setGraph, updateNodeData, setExecutionResult,
// duplicateSelectedNodes) so they don't need a full registry bootstrap.

const resetStore = () => {
  useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
};

describe('useGraphStore', () => {
  beforeEach(() => {
    resetStore();
  });

  it('setGraph replaces nodes and edges atomically', () => {
    const nodes: Node[] = [
      { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
      { id: 'b', type: 'custom', position: { x: 100, y: 0 }, data: { definitionType: 'encoding' } },
    ];
    const edges: Edge[] = [{ id: 'a-b', source: 'a', target: 'b' }];

    useGraphStore.getState().setGraph(nodes, edges);

    const state = useGraphStore.getState();
    expect(state.nodes).toHaveLength(2);
    expect(state.edges).toHaveLength(1);
    expect(state.nodes[0]?.id).toBe('a');
  });

  it('updateNodeData merges the patch onto the matching node only', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'a',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'imputation_node', strategy: 'mean' },
        },
        {
          id: 'b',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'encoding', method: 'onehot' },
        },
      ],
      [],
    );

    useGraphStore.getState().updateNodeData('a', { strategy: 'median', columns: ['x'] });

    const [a, b] = useGraphStore.getState().nodes;
    // patch merged, existing keys preserved
    expect(a?.data).toMatchObject({
      definitionType: 'imputation_node',
      strategy: 'median',
      columns: ['x'],
    });
    // sibling node untouched
    expect(b?.data).toMatchObject({ definitionType: 'encoding', method: 'onehot' });
  });

  it('duplicateSelectedNodes returns 0 and is a no-op when nothing is selected', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'a',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'imputation_node' },
        },
      ],
      [],
    );

    const cloned = useGraphStore.getState().duplicateSelectedNodes();
    expect(cloned).toBe(0);
    expect(useGraphStore.getState().nodes).toHaveLength(1);
  });

  it('duplicateSelectedNodes clones with a 32px offset and selects the clones', () => {
    useGraphStore.getState().setGraph(
      [
        {
          id: 'a',
          type: 'custom',
          position: { x: 100, y: 200 },
          selected: true,
          data: { definitionType: 'imputation_node', strategy: 'mean' },
        },
      ],
      [],
    );

    const cloned = useGraphStore.getState().duplicateSelectedNodes();
    expect(cloned).toBe(1);

    const nodes = useGraphStore.getState().nodes;
    expect(nodes).toHaveLength(2);

    const original = nodes.find((n) => n.id === 'a');
    const clone = nodes.find((n) => n.id !== 'a');
    expect(original?.selected).toBe(false); // original is now deselected
    expect(clone?.selected).toBe(true);
    expect(clone?.position).toEqual({ x: 132, y: 232 });
    // Cloned data is a fresh shallow copy — not the same object reference.
    expect(clone?.data).not.toBe(original?.data);
    expect(clone?.data).toMatchObject({ strategy: 'mean' });
  });

  it('setExecutionResult round-trips and clears with null', () => {
    const fakeResult = { nodes: {} } as unknown as Parameters<
      ReturnType<typeof useGraphStore.getState>['setExecutionResult']
    >[0];
    useGraphStore.getState().setExecutionResult(fakeResult);
    expect(useGraphStore.getState().executionResult).toBe(fakeResult);

    useGraphStore.getState().setExecutionResult(null);
    expect(useGraphStore.getState().executionResult).toBeNull();
  });
});

// Registry-dependent reducers — bootstrap the registry once so addNode,
// chainSiblings, and onConnect have node definitions to consult.
describe('useGraphStore — registry-dependent reducers', () => {
  beforeAll(() => {
    initializeRegistry();
  });
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
  });

  it('addNode appends a node with definitionType + default config and returns the new id', () => {
    const id = useGraphStore
      .getState()
      .addNode('imputation_node', { x: 50, y: 60 }, { columns: ['a'] });
    expect(id).toMatch(/^imputation_node-/);

    const node = useGraphStore.getState().nodes[0];
    expect(node?.id).toBe(id);
    expect(node?.position).toEqual({ x: 50, y: 60 });
    // The store stamps `definitionType` + `catalogType` from the type
    // arg and merges the registry default config plus the caller's
    // initialData on top.
    expect(node?.data).toMatchObject({
      definitionType: 'imputation_node',
      catalogType: 'imputation_node',
      columns: ['a'],
    });
  });

  it('addNode returns "" and is a no-op for an unknown type', () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const id = useGraphStore.getState().addNode('not_a_real_node', { x: 0, y: 0 });
    expect(id).toBe('');
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    errSpy.mockRestore();
  });

  it('onNodesChange applies React Flow position changes', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    useGraphStore.getState().onNodesChange([
      { id, type: 'position', position: { x: 200, y: 300 }, dragging: false },
    ]);
    expect(useGraphStore.getState().nodes[0]?.position).toEqual({ x: 200, y: 300 });
  });

  it('onNodesChange removes a node when given a remove change', () => {
    const id = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    useGraphStore.getState().onNodesChange([{ id, type: 'remove' }]);
    expect(useGraphStore.getState().nodes).toHaveLength(0);
  });

  it('onEdgesChange removes an edge', () => {
    useGraphStore.getState().setGraph(
      [],
      [{ id: 'e1', source: 'a', target: 'b' }],
    );
    useGraphStore.getState().onEdgesChange([{ id: 'e1', type: 'remove' }]);
    expect(useGraphStore.getState().edges).toHaveLength(0);
  });

  it('onConnect appends a valid edge between two non-model nodes', () => {
    const a = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    const b = useGraphStore.getState().addNode('encoding', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: a, target: b, sourceHandle: null, targetHandle: null });
    const edges = useGraphStore.getState().edges;
    expect(edges).toHaveLength(1);
    expect(edges[0]).toMatchObject({ source: a, target: b });
  });

  it('onConnect blocks model→model connections (training output into another training node)', () => {
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    const m1 = useGraphStore.getState().addNode('basic_training', { x: 0, y: 0 });
    const m2 = useGraphStore.getState().addNode('basic_training', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: m1, target: m2, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(alertSpy).toHaveBeenCalledOnce();
    alertSpy.mockRestore();
  });
});

// History (zundo temporal) integration — duplicate is a structural
// change so it should produce an undoable entry.
describe('useGraphStore — temporal undo/redo', () => {
  beforeAll(() => {
    initializeRegistry();
  });
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
    useGraphStore.temporal.getState().clear();
  });

  it('setGraph creates an undoable history entry', () => {
    const before = useGraphStore.temporal.getState().pastStates.length;
    useGraphStore.getState().setGraph(
      [{ id: 'x', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } }],
      [],
    );
    const after = useGraphStore.temporal.getState().pastStates.length;
    expect(after).toBeGreaterThan(before);
  });

  it('undo() reverts the most recent structural change', () => {
    useGraphStore.getState().setGraph(
      [{ id: 'x', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } }],
      [],
    );
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().nodes).toHaveLength(0);
  });
});
