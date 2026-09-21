import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react';
import type { Edge, Node } from '@xyflow/react';
import { RestoreSessionBanner } from './RestoreSessionBanner';
import { useRunControls } from './toolbar/_hooks/useRunControls';
import { initializeRegistry } from '../../core/registry/init';
import { registry } from '../../core/registry/NodeRegistry';
import { useGraphStore } from '../../core/store/useGraphStore';
import { confirmConnection } from '../../core/store/graphStore/connectionPolicy';
import { useViewStore } from '../../core/store/useViewStore';
import { saveCanvasSnapshot } from '../../core/utils/canvasPersistence';
import { connectionIssue } from '../../core/utils/connectionValidation';
import { convertGraphToPipelineConfig } from '../../core/utils/pipelineConverter';
import { incomingModelNodes } from '../../modules/nodes/modeling/ensembleSettings/connectedModels';
import { runPipelinePreview } from '../../core/api/client';
import { jobsApi } from '../../core/api/jobs';

vi.mock('../../core/api/client', () => ({ runPipelinePreview: vi.fn() }));
vi.mock('../../core/api/jobs', () => ({ jobsApi: { runPipeline: vi.fn() } }));

/** Use real definitions for active nodes and the saved legacy model's original type. */
function savedNode(id: string, definitionType: string, data: Record<string, unknown>): Node {
  return {
    id, type: 'custom', position: { x: 0, y: 0 },
    data: { ...registry.get(definitionType)?.getDefaultConfig() as object, ...data, definitionType },
  };
}

describe('legacy training restore contract (QW-111)', () => {
  beforeAll(() => initializeRegistry());
  beforeEach(() => {
    window.localStorage.clear();
    vi.clearAllMocks();
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null, lastRunError: null });
    useViewStore.setState({ readOnlyOverride: 'off', leakageNotice: null });
  });
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });

  it('restores legacy data for recovery but rejects ports and execution despite converter compatibility', async () => {
    // Converter aliases must never silently grant an unregistered saved node permission to run.
    const nodes = [
      savedNode('dataset', 'dataset_node', { datasetId: 'ds-1' }),
      savedNode('legacy', 'training', {
        task: 'classification', model_type: 'random_forest_classifier',
        target_column: 'label', hyperparameters: { n_estimators: 17 },
      }),
      savedNode('ensemble', 'EnsembleNode', { target_column: 'label' }),
    ];
    const edges: Edge[] = [
      { id: 'data-model', source: 'dataset', sourceHandle: 'data', target: 'legacy', targetHandle: 'in' },
      { id: 'model-ensemble', source: 'legacy', sourceHandle: 'model', target: 'ensemble', targetHandle: 'in' },
    ];
    saveCanvasSnapshot(nodes, edges);
    render(<RestoreSessionBanner />);
    fireEvent.click(screen.getByRole('button', { name: 'Restore' }));
    const restored = useGraphStore.getState();
    expect(restored.nodes).toEqual(nodes);
    expect(restored.edges).toEqual(edges);
    expect(registry.get('training')).toBeUndefined();

    const connection = { source: 'legacy', sourceHandle: 'model', target: 'ensemble', targetHandle: 'in' };
    const portError = 'Choose an output and an input on registered nodes. Data sources have no input.';
    expect(connectionIssue(restored.nodes, [], connection)).toBe(portError);
    expect(confirmConnection(connection, restored.nodes, [])).toBe(false);
    expect(connectionIssue(restored.nodes, [], { ...connection, source: 'dataset', sourceHandle: 'data', target: 'legacy' })).toBe(portError);
    expect(restored.validateGraph()).toContainEqual({
      nodeId: 'legacy', nodeLabel: 'Training', category: 'configuration',
      message: 'Refresh or re-add Training so the canvas knows how to validate it.',
    });

    // Both legacy compatibility readers can inspect the recipe; neither owns run authorization.
    expect(incomingModelNodes('ensemble', restored.nodes, restored.edges).map(node => node.id)).toEqual(['legacy']);
    const converted = convertGraphToPipelineConfig(restored.nodes, restored.edges);
    expect(converted.nodes.map(node => node.node_id)).toEqual(['dataset', 'ensemble']);
    expect(converted.nodes[1]).toMatchObject({
      inputs: ['dataset'], step_type: 'training',
      params: { hyperparameters: {
        base_estimators: ['random_forest'], base_estimator_params: { random_forest: { n_estimators: 17 } },
      } },
    });
    const { result } = renderHook(() => useRunControls());
    expect(result.current.canRunPreview).toBe(false);
    await act(async () => { await result.current.handleRun(); await result.current.handleRunAll(); });
    expect(runPipelinePreview).not.toHaveBeenCalled();
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
  });

  it.each(['classification', 'regression', 'text_classification'])('keeps registered %s model ports available', (type) => {
    // Rejecting legacy recovery data must not loosen or disable supported model-to-ensemble wiring.
    const nodes = [savedNode('model', type, {}), savedNode('ensemble', 'EnsembleNode', {})];
    expect(registry.get(type)).toBeDefined();
    expect(connectionIssue(nodes, [], {
      source: 'model', sourceHandle: 'model', target: 'ensemble', targetHandle: 'in',
    })).toBeNull();
  });
});
