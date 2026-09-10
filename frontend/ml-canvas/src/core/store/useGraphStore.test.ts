import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { collectGraphValidationIssues, useGraphStore } from './useGraphStore';
import { initializeRegistry } from '../registry/init';
import { registry } from '../registry/NodeRegistry';
import { useViewStore } from './useViewStore';
import { toast } from '../toast';

// Pure-store smoke tests. These exercise the actions that don't depend on
// the NodeRegistry (setGraph, updateNodeData, setExecutionResult,
// duplicateSelectedNodes) so they don't need a full registry bootstrap.

const resetStore = () => {
  useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
};

/** Build real registered nodes so characterization exercises the public graph boundary. */
function graphNode(id: string, definitionType: string, data: Record<string, unknown> = {}): Node {
  return {
    id, type: 'custom', position: { x: 0, y: 0 },
    data: { ...registry.get(definitionType)?.getDefaultConfig() as object, definitionType, ...data },
  };
}

const xyWarning = 'Warning: X/Y Split without a prior Train-Test Split.\n\n' +
  'This means 100% of data will be used (possible data leakage).\n\n' +
  'Click OK to connect anyway, or Cancel to abort.';
const ensembleWarning = 'Warning: this model comes from a different dataset than the ensemble\'s ' +
  'other inputs.\n\nThe ensemble re-fits every base learner on a single dataset, so mixing ' +
  'models trained on unrelated data is usually a wiring mistake.\n\n' +
  'Click OK to connect anyway, or Cancel to abort.';
const trainingWarning = 'This training node will receive 2 inputs.\n\nYou have two options:\n' +
  '  • MERGE (default): Inputs are auto-merged into one dataset before training.\n' +
  '  • PARALLEL: Each input runs as a separate experiment.\n' +
  '    → To use parallel mode, connect each path to its OWN training node.\n\n' +
  'Click OK to connect (merge mode), or Cancel to abort.';
const processingWarning = 'This node will receive 2 inputs.\n\n' +
  'Inputs are merged into one dataset. Each column keeps the value of ' +
  'whichever branch changed it, so branches editing different columns ' +
  'are combined without loss.\n\nIf two branches change the SAME column to different values, one of ' +
  'them is discarded. The run Results panel reports any such conflict ' +
  'and lets you choose which branch wins.\n\n' +
  'For strictly sequential transformations, chain the nodes linearly instead.\n\n' +
  'Click OK to connect (merge), or Cancel to abort.';

describe('graph validation public characterization', () => {
  beforeAll(() => { initializeRegistry(); });
  afterEach(() => { vi.restoreAllMocks(); });

  it('excludes preview nodes and their wires before checking required connections', () => {
    // Preview wiring must neither hide missing inputs nor satisfy a Dataset output.
    const nodes = [graphNode('dataset', 'dataset_node', { datasetId: 'ds' }),
      graphNode('preview', 'data_preview'), graphNode('split', 'TrainTestSplitter')];
    const issues = collectGraphValidationIssues(nodes, [
      { id: 'a', source: 'dataset', target: 'preview' },
      { id: 'b', source: 'preview', target: 'split' },
      { id: 'c', source: 'preview', target: 'preview' },
    ]);
    expect(issues).toEqual([
      { nodeId: 'dataset', nodeLabel: 'Dataset', category: 'connection', message: 'Connect a downstream node to this Dataset before running preview.' },
      { nodeId: 'split', nodeLabel: 'Train-Test Split', category: 'connection', message: 'Connect an upstream node to Train-Test Split before running preview.' },
    ]);
  });

  it.each([
    [{ label: 'Custom', title: 'Title' }, 'Custom'],
    [{ label: '', title: 'Title' }, 'Title'],
    [{ label: 42, title: '' }, 'Unknown Kind'],
  ])('preserves unknown-definition label fallback %j', (data, label) => {
    // Missing definitions report exactly one configuration issue with no field.
    expect(collectGraphValidationIssues([graphNode('unknown', 'unknown_kind', data)], [])).toEqual([
      { nodeId: 'unknown', nodeLabel: label, category: 'configuration', message: `Refresh or re-add ${label} so the canvas knows how to validate it.` },
    ]);
  });

  it('falls back to the node ID when the definition type is absent', () => {
    // Legacy saved nodes without a definition still need an actionable label.
    expect(collectGraphValidationIssues([{ id: 'legacy', position: { x: 0, y: 0 }, data: {} }], [])[0]?.nodeLabel).toBe('legacy');
  });

  it.each([
    [{ isValid: false, message: 'bad setting', field: 'value' }, ': bad setting', { field: 'value' }],
    [{ isValid: false, message: '', field: '' }, '.', {}],
  ])('keeps configuration before connection and optional field semantics %j', (validation, suffix, field) => {
    // Empty validation text and fields must retain the original fallback punctuation and shape.
    vi.spyOn(registry.get('TrainTestSplitter')!, 'validate').mockReturnValue(validation);
    expect(collectGraphValidationIssues([graphNode('split', 'TrainTestSplitter', { label: 'Split' })], [])).toEqual([
      { nodeId: 'split', nodeLabel: 'Split', category: 'configuration', message: `Fix the Split settings before running preview${suffix}`, ...field },
      { nodeId: 'split', nodeLabel: 'Split', category: 'connection', message: 'Connect an upstream node to Split before running preview.' },
    ]);
  });

  it.each([undefined, '', 42, 'ds'])('requires outgoing Dataset wiring only for a nonempty string ID: %s', datasetId => {
    // Dataset validation and output eligibility intentionally use different truthiness rules.
    const issues = collectGraphValidationIssues([graphNode('ds', 'dataset_node', { datasetId })], []);
    expect(issues.filter(issue => issue.category === 'connection')).toHaveLength(datasetId === 'ds' ? 1 : 0);
  });

  it('appends leakage then cycles after per-node issues with exact labels and ordering', () => {
    // Bulk-loaded graphs must surface all blocking categories in a stable order.
    const nodes = [graphNode('unknown', 'unknown_kind'), graphNode('ds', 'dataset_node', { datasetId: 'ds' }),
      graphNode('scale', 'scale_numeric_features', { label: 'Scale', columns: ['a'] }),
      graphNode('split', 'TrainTestSplitter', { title: 'Holdout' }),
      graphNode('a', 'TrainTestSplitter', { label: 'A' }), graphNode('b', 'TrainTestSplitter', { label: 'B' })];
    const edges = [{ id: '1', source: 'ds', target: 'scale' }, { id: '2', source: 'scale', target: 'split' },
      { id: '3', source: 'a', target: 'b' }, { id: '4', source: 'b', target: 'a' },
      { id: '5', source: 'ds', target: 'a' }];
    useGraphStore.getState().setGraph(nodes, edges);
    const issues = collectGraphValidationIssues(nodes, edges);
    expect(useGraphStore.getState().validateGraph()).toEqual(issues);
    expect(issues).toEqual([
      { nodeId: 'unknown', nodeLabel: 'Unknown Kind', category: 'configuration', message: 'Refresh or re-add Unknown Kind so the canvas knows how to validate it.' },
      { nodeId: 'scale', nodeLabel: 'Scale', category: 'leakage', message: 'Move Scale after Holdout so it only fits on training data.' },
      { nodeId: 'a', nodeLabel: 'A', category: 'cycle', message: 'Cycle detected: A -> B feed back into each other. Remove one of these connections so the pipeline flows in one direction.' },
    ]);
  });
});

describe('connection policy public characterization', () => {
  beforeAll(() => { initializeRegistry(); });
  beforeEach(() => {
    resetStore();
    useGraphStore.temporal.getState().clear();
    useViewStore.setState({ readOnlyOverride: 'off' });
  });
  afterEach(() => { vi.restoreAllMocks(); useViewStore.setState({ readOnlyOverride: 'auto' }); });

  it.each(['manual', 'next'] as const)('rejects invalid %s wiring before any warning or mutation', mode => {
    // A rejected edge must not show downstream warnings or create an orphan next node.
    useGraphStore.getState().setGraph([graphNode('xy', 'feature_target_split')], []);
    useGraphStore.temporal.getState().clear();
    const before = useGraphStore.getState();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const error = vi.spyOn(toast, 'error').mockImplementation(() => {});
    if (mode === 'manual') before.onConnect({ source: 'xy', target: 'xy', sourceHandle: 'X', targetHandle: 'in' });
    else expect(before.addConnectedNode('xy', 'missing', 'imputation_node', 'in', { x: 10, y: 0 })).toBe('');
    expect(error).toHaveBeenCalledOnce();
    expect(confirm).not.toHaveBeenCalled();
    expect(useGraphStore.getState()).toBe(before);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
  });

  it.each(['manual', 'next'] as const)('synchronously accepts or cancels the exact X/Y warning for %s insertion', mode => {
    // Confirmation resolves before either action publishes graph or history changes.
    useGraphStore.getState().setGraph([graphNode('xy', 'feature_target_split'), graphNode('target', 'imputation_node')], []);
    useGraphStore.temporal.getState().clear();
    const before = useGraphStore.getState();
    const confirm = vi.spyOn(window, 'confirm').mockImplementation(() => {
      expect(useGraphStore.getState()).toBe(before);
      return false;
    });
    const connect = () => mode === 'manual'
      ? before.onConnect({ source: 'xy', target: 'target', sourceHandle: 'X', targetHandle: 'in' })
      : before.addConnectedNode('xy', 'X', 'imputation_node', 'in', { x: 10, y: 0 });
    connect();
    expect(confirm).toHaveBeenCalledExactlyOnceWith(xyWarning);
    expect(useGraphStore.getState()).toBe(before);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
    confirm.mockReturnValue(true);
    connect();
    expect(useGraphStore.getState().edges).toHaveLength(1);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().nodes).toBe(before.nodes);
    expect(useGraphStore.getState().edges).toBe(before.edges);
    useGraphStore.temporal.getState().redo();
    expect(useGraphStore.getState().edges).toHaveLength(1);
  });

  it.each(['ancestor', 'target'] as const)('suppresses the X/Y warning for a TrainTestSplitter %s', location => {
    // Split discovery includes the target as well as upstream ancestry.
    const nodes = [graphNode('xy', 'feature_target_split'), graphNode('split', 'TrainTestSplitter'), graphNode('target', 'imputation_node')];
    const edges = location === 'ancestor' ? [{ id: 'up', source: 'split', target: 'xy' }] : [];
    useGraphStore.getState().setGraph(nodes, edges);
    const confirm = vi.spyOn(window, 'confirm');
    useGraphStore.getState().onConnect({ source: 'xy', target: location === 'target' ? 'split' : 'target', sourceHandle: 'X', targetHandle: 'in' });
    expect(confirm).not.toHaveBeenCalled();
    expect(useGraphStore.getState().edges).toHaveLength(edges.length + 1);
  });

  it.each(['classification', 'imputation_node'])('preserves warning order and cancellation for %s fan-in', targetType => {
    // X/Y leakage confirmation precedes the distinct-source merge confirmation.
    useGraphStore.getState().setGraph([graphNode('xy', 'feature_target_split'), graphNode('other', 'dataset_node'), graphNode('target', targetType)],
      [{ id: 'existing', source: 'other', target: 'target', targetHandle: 'in' }]);
    useGraphStore.temporal.getState().clear();
    const before = useGraphStore.getState();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(true).mockReturnValueOnce(false);
    before.onConnect({ source: 'xy', target: 'target', sourceHandle: 'X', targetHandle: 'in' });
    expect(confirm.mock.calls).toEqual([[xyWarning], [targetType === 'classification' ? trainingWarning : processingWarning]]);
    expect(useGraphStore.getState()).toBe(before);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
    confirm.mockClear().mockReturnValueOnce(false);
    before.onConnect({ source: 'xy', target: 'target', sourceHandle: 'X', targetHandle: 'in' });
    expect(confirm.mock.calls).toEqual([[xyWarning]]);
    confirm.mockClear().mockReturnValue(true);
    before.onConnect({ source: 'xy', target: 'target', sourceHandle: 'X', targetHandle: 'in' });
    expect(useGraphStore.getState().edges).toHaveLength(2);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
  });

  it.each(['data_preview', 'EnsembleNode'])('suppresses fan-in prompts for %s', type => {
    // These consumers use their own input contract instead of the merge contract.
    useGraphStore.getState().setGraph([graphNode('a', 'dataset_node'), graphNode('b', 'dataset_node'), graphNode('target', type)],
      [{ id: 'first', source: 'a', target: 'target' }]);
    const confirm = vi.spyOn(window, 'confirm');
    useGraphStore.getState().onConnect({ source: 'b', target: 'target', sourceHandle: null, targetHandle: null });
    expect(confirm).not.toHaveBeenCalled();
    expect(useGraphStore.getState().edges).toHaveLength(2);
  });

  it('counts distinct sources despite duplicate saved handles and rejects duplicate split wiring first', () => {
    // Legacy parallel handles count as one upstream branch, not multiple experiments.
    const nodes = [graphNode('a', 'TrainTestSplitter'), graphNode('b', 'dataset_node'), graphNode('target', 'classification')];
    useGraphStore.setState({ nodes, edges: ['train', 'test'].map(handle => ({ id: handle, source: 'a', target: 'target', sourceHandle: handle, targetHandle: 'in' })) });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const error = vi.spyOn(toast, 'error').mockImplementation(() => {});
    useGraphStore.getState().onConnect({ source: 'a', target: 'target', sourceHandle: 'test', targetHandle: 'in' });
    expect(error).toHaveBeenCalledExactlyOnceWith('Invalid connection', 'These ports are already connected. Choose another input or remove the existing connection.');
    expect(confirm).not.toHaveBeenCalled();
    useGraphStore.getState().onConnect({ source: 'b', target: 'target', sourceHandle: null, targetHandle: 'in' });
    expect(confirm).toHaveBeenCalledExactlyOnceWith(trainingWarning);
    expect(useGraphStore.getState().edges).toHaveLength(3);
  });

  it.each(['disjoint', 'shared', 'unknown'] as const)('warns only for disjoint known ensemble dataset ancestry: %s', lineage => {
    // Ensemble inputs are model specifications; only a proven lineage mismatch prompts.
    const nodes = [graphNode('ds1', 'dataset_node'), graphNode('ds2', 'dataset_node'),
      graphNode('model', 'classification'), graphNode('ensemble', 'EnsembleNode')];
    const edges = [{ id: 'existing', source: 'ds1', target: 'ensemble' }];
    if (lineage !== 'unknown') edges.push({ id: 'lineage', source: lineage === 'shared' ? 'ds1' : 'ds2', target: 'model' });
    useGraphStore.getState().setGraph(nodes, edges);
    useGraphStore.temporal.getState().clear();
    const before = useGraphStore.getState();
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    const connect = () => before.onConnect({ source: 'model', target: 'ensemble', sourceHandle: null, targetHandle: null });
    connect();
    if (lineage === 'disjoint') {
      expect(confirm).toHaveBeenCalledExactlyOnceWith(ensembleWarning);
      expect(useGraphStore.getState()).toBe(before);
      expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
      confirm.mockReturnValue(true);
      connect();
    } else expect(confirm).not.toHaveBeenCalled();
    expect(useGraphStore.getState().edges).toHaveLength(edges.length + 1);
  });
});

describe('history equality public characterization', () => {
  beforeEach(() => {
    useGraphStore.setState({ nodes: [graphNode('a', 'dataset_node'), graphNode('b', 'dataset_node')], edges: [], executionResult: null });
    useGraphStore.temporal.getState().clear();
  });

  it('tracks a new edge array even if its contents are unchanged', () => {
    // Edge identity is the history contract, unlike node selection equality.
    const before = useGraphStore.getState().edges;
    useGraphStore.setState({ edges: [...before] });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().edges).toBe(before);
    useGraphStore.temporal.getState().redo();
    expect(useGraphStore.getState().edges).not.toBe(before);
  });

  it.each(['reorder', 'data', 'type', 'id', 'length', 'x', 'y'])('records and restores node %s changes', change => {
    // Equality must retain each structural distinction used by undo/redo.
    const before = useGraphStore.getState().nodes;
    const first = before[0]!;
    const changes: Record<string, Node[]> = {
      reorder: [...before].reverse(), data: [{ ...first, data: { ...first.data } }, before[1]!],
      type: [{ ...first, type: 'different' }, before[1]!], id: [{ ...first, id: 'other' }, before[1]!],
      length: [first], x: [{ ...first, position: { x: 1, y: 0 } }, before[1]!],
      y: [{ ...first, position: { x: 0, y: 1 } }, before[1]!],
    };
    useGraphStore.setState({ nodes: changes[change]! });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().nodes).toBe(before);
    useGraphStore.temporal.getState().redo();
    expect(useGraphStore.getState().nodes).toBe(changes[change]);
  });

  it('ignores node-array identity, equivalent replacements, selection, and execution results', () => {
    // UI-only state and backend preview payloads must stay outside graph history.
    const nodes = useGraphStore.getState().nodes;
    useGraphStore.setState({ nodes });
    useGraphStore.setState({ nodes: nodes.map(node => ({ ...node, position: { ...node.position } })) });
    useGraphStore.getState().selectNode('a');
    useGraphStore.getState().setExecutionResult({ nodes: {} } as never);
    useGraphStore.getState().updateNodeData('a', { changed: true });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    expect(Object.keys(useGraphStore.temporal.getState().pastStates[0]!)).toEqual(['nodes', 'edges']);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().executionResult).toEqual({ nodes: {} });
    expect(useGraphStore.getState().nodes[0]?.selected).toBe(true);
  });

  it('currently ignores both in-progress drag positions and the drag-end transition', () => {
    // Pin the existing either-side-dragging rule, including its drag-end behavior.
    const onNodesChange = useGraphStore.getState().onNodesChange;
    onNodesChange([{ id: 'a', type: 'position', position: { x: 10, y: 20 }, dragging: true }]);
    onNodesChange([{ id: 'a', type: 'position', position: { x: 30, y: 40 }, dragging: true }]);
    onNodesChange([{ id: 'a', type: 'position', position: { x: 50, y: 60 }, dragging: false }]);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
    onNodesChange([{ id: 'a', type: 'position', position: { x: 70, y: 80 }, dragging: false }]);
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().nodes[0]?.position).toEqual({ x: 50, y: 60 });
  });

  it('still records data changes on a dragging node', () => {
    // Drag suppression applies only after identity, data, and type comparisons.
    const before = useGraphStore.getState().nodes;
    const node = before[0]!;
    const changedData = { ...node.data };
    const next = [{ ...node, dragging: true, data: changedData }, before[1]!];
    useGraphStore.setState({ nodes: next });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(1);
    expect(useGraphStore.getState().nodes[0]?.data).toBe(changedData);
    useGraphStore.temporal.getState().undo();
    expect(useGraphStore.getState().nodes).toBe(before);
    expect(useGraphStore.getState().nodes[0]?.data).toBe(node.data);
    useGraphStore.temporal.getState().redo();
    expect(useGraphStore.getState().nodes).toBe(next);
    expect(useGraphStore.getState().nodes[0]?.data).toBe(changedData);
  });

  it('caps history at the latest 100 structural changes', () => {
    // Long editing sessions must retain the existing bounded snapshot policy.
    for (let index = 1; index <= 105; index++) useGraphStore.getState().updateNodeData('a', { index });
    expect(useGraphStore.temporal.getState().pastStates).toHaveLength(100);
    useGraphStore.temporal.getState().undo(100);
    expect(useGraphStore.getState().nodes[0]?.data.index).toBe(5);
    useGraphStore.temporal.getState().redo(100);
    expect(useGraphStore.getState().nodes[0]?.data.index).toBe(105);
  });
});

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

  it('validateGraph returns structured issues for configuration, connection, and leakage failures', () => {
    initializeRegistry();
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset-missing',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node' },
        },
        {
          id: 'orphan-encoding',
          type: 'custom',
          position: { x: 120, y: 0 },
          data: { definitionType: 'encoding', method: 'label', columns: ['status'] },
        },
        {
          id: 'leak-dataset',
          type: 'custom',
          position: { x: 0, y: 160 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'leaky-imputer',
          type: 'custom',
          position: { x: 120, y: 160 },
          data: {
            definitionType: 'imputation_node',
            columns: ['feature_a'],
            method: 'simple',
            strategy: 'mean',
          },
        },
        {
          id: 'splitter',
          type: 'custom',
          position: { x: 240, y: 160 },
          data: {
            definitionType: 'TrainTestSplitter',
            test_size: 0.2,
            validation_size: 0,
            random_state: 42,
            stratify: false,
            shuffle: true,
          },
        },
      ],
      [
        { id: 'e1', source: 'leak-dataset', target: 'leaky-imputer' },
        { id: 'e2', source: 'leaky-imputer', target: 'splitter' },
      ],
    );

    const issues = useGraphStore.getState().validateGraph();
    expect(issues).toHaveLength(3);
    expect(issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          nodeId: 'dataset-missing',
          nodeLabel: 'Dataset',
          category: 'configuration',
        }),
        expect.objectContaining({
          nodeId: 'orphan-encoding',
          nodeLabel: 'Encoding',
          category: 'connection',
        }),
        expect.objectContaining({
          nodeId: 'leaky-imputer',
          nodeLabel: 'Imputation',
          category: 'leakage',
        }),
      ]),
    );
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

  it('onConnect blocks model→model connections (training output into another training node)', async () => {
    const toastModule = await import('../toast');
    const toastSpy = vi.spyOn(toastModule.toast, 'error').mockImplementation(() => {});
    const m1 = useGraphStore.getState().addNode('classification', { x: 0, y: 0 });
    const m2 = useGraphStore.getState().addNode('classification', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: m1, target: m2, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(toastSpy).toHaveBeenCalledOnce();
    toastSpy.mockRestore();
  });

  it('onConnect blocks a connection that closes a multi-hop cycle', async () => {
    const toastModule = await import('../toast');
    const toastSpy = vi.spyOn(toastModule.toast, 'error').mockImplementation(() => {});
    const a = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    const b = useGraphStore.getState().addNode('encoding', { x: 100, y: 0 });
    const c = useGraphStore.getState().addNode('scale_numeric_features', { x: 200, y: 0 });
    const connect = useGraphStore.getState().onConnect;
    connect({ source: a, target: b, sourceHandle: null, targetHandle: null });
    connect({ source: b, target: c, sourceHandle: null, targetHandle: null });
    // c → a would close the loop a → b → c → a
    connect({ source: c, target: a, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(2);
    expect(toastSpy).toHaveBeenCalledOnce();
    toastSpy.mockRestore();
  });

  it('onConnect blocks a direct self-connection', async () => {
    const toastModule = await import('../toast');
    const toastSpy = vi.spyOn(toastModule.toast, 'error').mockImplementation(() => {});
    const a = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    useGraphStore.getState().onConnect({ source: a, target: a, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(toastSpy).toHaveBeenCalledOnce();
    toastSpy.mockRestore();
  });

  it('onConnect allows a diamond shape (shared ancestor is not a cycle)', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const a = useGraphStore.getState().addNode('imputation_node', { x: 0, y: 0 });
    const b = useGraphStore.getState().addNode('encoding', { x: 100, y: 0 });
    const c = useGraphStore.getState().addNode('scale_numeric_features', { x: 100, y: 100 });
    const d = useGraphStore.getState().addNode('TrainTestSplitter', { x: 200, y: 50 });
    const connect = useGraphStore.getState().onConnect;
    connect({ source: a, target: b, sourceHandle: null, targetHandle: null });
    connect({ source: a, target: c, sourceHandle: null, targetHandle: null });
    connect({ source: b, target: d, sourceHandle: null, targetHandle: null });
    // Second fan-in source: the existing merge confirm fires (accepted here);
    // the cycle guard must NOT block it — b and c share ancestor a, no loop.
    connect({ source: c, target: d, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(4);
    confirmSpy.mockRestore();
  });

  it('onConnect blocks model output feeding back into a preprocessing node', async () => {
    const toastModule = await import('../toast');
    const toastSpy = vi.spyOn(toastModule.toast, 'error').mockImplementation(() => {});
    const model = useGraphStore.getState().addNode('classification', { x: 0, y: 0 });
    const imputer = useGraphStore.getState().addNode('imputation_node', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: model, target: imputer, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(toastSpy).toHaveBeenCalledOnce();
    toastSpy.mockRestore();
  });

  it('onConnect blocks model output into a splitter node', async () => {
    const toastModule = await import('../toast');
    const toastSpy = vi.spyOn(toastModule.toast, 'error').mockImplementation(() => {});
    const model = useGraphStore.getState().addNode('regression', { x: 0, y: 0 });
    const splitter = useGraphStore.getState().addNode('TrainTestSplitter', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: model, target: splitter, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(0);
    expect(toastSpy).toHaveBeenCalledOnce();
    toastSpy.mockRestore();
  });

  it('onConnect still allows model output into an EnsembleNode', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const model = useGraphStore.getState().addNode('classification', { x: 0, y: 0 });
    const ensemble = useGraphStore.getState().addNode('EnsembleNode', { x: 100, y: 0 });
    useGraphStore.getState().onConnect({ source: model, target: ensemble, sourceHandle: null, targetHandle: null });
    expect(useGraphStore.getState().edges).toHaveLength(1);
    confirmSpy.mockRestore();
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
