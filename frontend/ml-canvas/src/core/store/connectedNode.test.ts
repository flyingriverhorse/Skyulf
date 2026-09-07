import { beforeAll, beforeEach, afterEach, expect, it, vi } from 'vitest';
import { initializeRegistry } from '../registry/init';
import { useGraphStore } from './useGraphStore';
import { useViewStore } from './useViewStore';

beforeAll(() => { initializeRegistry(); });
beforeEach(() => {
  useGraphStore.setState({ nodes: [], edges: [] });
  useGraphStore.temporal.getState().clear();
  useViewStore.setState({ readOnlyOverride: 'off' });
});
afterEach(() => { vi.restoreAllMocks(); useViewStore.setState({ readOnlyOverride: 'auto' }); });

it('adds and connects a split output in one undoable update', () => {
  // Undo must remove both the new node and wire, preserving the original split.
  const source = useGraphStore.getState().addNode('TrainTestSplitter', { x: 0, y: 0 });
  useGraphStore.temporal.getState().clear();
  const id = useGraphStore.getState().addConnectedNode(source, 'test', 'imputation_node', 'in', { x: 350, y: 0 });
  expect(id).not.toBe('');
  expect(useGraphStore.getState().nodes).toHaveLength(2);
  expect(useGraphStore.getState().edges).toEqual([expect.objectContaining({ source, target: id, sourceHandle: 'train', targetHandle: 'in' })]);
  expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
  useGraphStore.temporal.getState().undo();
  expect(useGraphStore.getState().nodes.map(node => node.id)).toEqual([source]);
  expect(useGraphStore.getState().edges).toHaveLength(0);
  useGraphStore.temporal.getState().redo();
  expect(useGraphStore.getState().nodes.find(node => node.id === id)?.selected).toBe(true);
  expect(useGraphStore.getState().edges).toHaveLength(1);
});

it('manual wiring groups split outputs and loading merges old parallel handle wires', () => {
  // Split sets are one logical backend input and must have one canvas delete/undo action.
  const source = useGraphStore.getState().addNode('TrainTestSplitter', { x: 0, y: 0 }, { validation_size: 0.1 });
  const target = useGraphStore.getState().addNode('imputation_node', { x: 350, y: 0 });
  useGraphStore.getState().onConnect({ source, target, sourceHandle: 'validation', targetHandle: 'in' });
  expect(useGraphStore.getState().edges).toEqual([expect.objectContaining({ sourceHandle: 'train' })]);
  const nodes = useGraphStore.getState().nodes;
  useGraphStore.getState().setGraph(nodes, ['train', 'validation', 'test'].map(handle => ({ id: handle, source, target, sourceHandle: handle, targetHandle: 'in' })));
  expect(useGraphStore.getState().edges).toHaveLength(1);
  expect(useGraphStore.getState().edges[0]?.id).toBe('train');
});

it('cancelling an existing leakage warning leaves no node or history entry', () => {
  // The picker must preserve the same warning as manually wiring X/Y Split.
  const source = useGraphStore.getState().addNode('feature_target_split', { x: 0, y: 0 });
  const output = 'X';
  useGraphStore.temporal.getState().clear();
  const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
  const id = useGraphStore.getState().addConnectedNode(source, output, 'imputation_node', 'in', { x: 350, y: 0 });
  expect(confirm).toHaveBeenCalledOnce();
  expect(id).toBe('');
  expect(useGraphStore.getState().nodes).toHaveLength(1);
  expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
});

it('rejects stale endpoints and read-only insertion without an orphan node', () => {
  // A picker opened before a graph or mode change must recheck mutation eligibility.
  const source = useGraphStore.getState().addNode('classification', { x: 0, y: 0 });
  expect(useGraphStore.getState().addConnectedNode(source, 'model', 'imputation_node', 'in', { x: 350, y: 0 })).toBe('');
  expect(useGraphStore.getState().addConnectedNode('deleted', 'out', 'imputation_node', 'in', { x: 350, y: 0 })).toBe('');
  useViewStore.setState({ readOnlyOverride: 'on' });
  expect(useGraphStore.getState().addConnectedNode(source, 'model', 'EnsembleNode', 'in', { x: 350, y: 0 })).toBe('');
  expect(useGraphStore.getState().nodes).toHaveLength(1);
  expect(useGraphStore.getState().edges).toHaveLength(0);
});
