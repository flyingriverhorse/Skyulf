import { useEffect, useMemo, useRef, useState } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../../../../core/store/useGraphStore';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { useNotificationsStore } from '../../../../core/store/useNotificationsStore';
import { useNodeInspectionStore } from '../../../../core/store/useNodeInspectionStore';
import { buildPreviewConfiguration } from '../../../../core/utils/previewConfiguration';
import { jobsApi, type RunPipelineResponse } from '../../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../../core/utils/pipelineConverter';
import { RUN_PREVIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import { registry } from '../../../../core/registry/NodeRegistry';
import { nodeDisplayNames } from '../../../../core/utils/nodeDisplayNames';
import { getReadOnlyMode } from '../../../../core/hooks/useReadOnlyMode';
import { getLeakageErrorMessage, graphSemanticSignature } from '../../../../core/utils/leakageFeedback';

const TRAINING_TYPES = new Set(['training', 'classification', 'regression', 'text_classification']);

export interface RunControls {
  isRunning: boolean;
  isRunningAll: boolean;
  canRunPreview: boolean;
  hasMultipleBranches: boolean;
  experimentModels: { id: string; name: string; model: string; action: string }[];
  experimentBlockReason: string;
  handleRun: () => Promise<void>;
  handleRunAll: () => Promise<void>;
}

/** Computes the toolbar's run availability and submit handlers. */
export function useRunControls(): RunControls {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const setExecutionResult = useGraphStore((s) => s.setExecutionResult);
  const setLastRunError = useGraphStore((s) => s.setLastRunError);
  const { toggleDrawer, setActiveParallelRun, startPolling } = useJobStore();
  const { setResultsPanelExpanded, setResultsPanelDismissed, setLeakageNotice } = useViewStore();

  const isRunning = useNodeInspectionStore(state => state.isLoading);
  const [isRunningAll, setIsRunningAll] = useState(false);
  const previewPending = useRef(false);
  const experimentsPending = useRef(false);
  const notifyPreview = (message: string) => useNotificationsStore.getState()
    .upsertExecution('canvas-preview', message, { type: 'preview' }, 'error');
  const notifyExperimentError = (message: string) => useNotificationsStore.getState()
    .upsertExecution('canvas-experiments-error', message, { type: 'preview' }, 'error');

  const validationIssues = useMemo(() => collectGraphValidationIssues(nodes, edges), [nodes, edges]);
  const experimentModels = useMemo(() => {
    const names = nodeDisplayNames(nodes);
    return nodes.filter(node => edges.some(edge => edge.target === node.id)
      && registry.get(String(node.data.definitionType))?.outputs.some(port => port.type === 'model'))
      .map(node => ({ id: node.id, name: names.get(node.id)!,
        model: String(node.data.model_type || 'Select a model').replace(/_/g, ' '),
        action: node.data.run_mode === 'advanced' ? 'Tune' : 'Train',
      }));
  }, [nodes, edges]);
  const experimentBlockReason = validationIssues.length > 0
    ? `Fix ${validationIssues.length} validation issue${validationIssues.length === 1 ? '' : 's'} first. ${validationIssues[0]!.message}`
    : experimentModels.length === 0 ? 'Connect a model to a dataset pipeline first.' : '';

  const canRunPreview = useMemo(() => {
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    if (!datasetNode) return false;
    const datasetId = datasetNode.data.datasetId as string | undefined;
    if (!datasetId) return false;
    if (!edges.some((e) => e.source === datasetNode.id)) return false;
    return validationIssues.length === 0;
  }, [nodes, edges, validationIssues]);

  const hasMultipleBranches = useMemo(() => {
    const trainingNodes = nodes.filter(
      (n) =>
        TRAINING_TYPES.has(n.data.definitionType as string) &&
        edges.some((e) => e.target === n.id),
    );
    if (trainingNodes.length < 2) return false;
    const parentSets = trainingNodes.map(
      (tn) => new Set(edges.filter((e) => e.target === tn.id).map((e) => e.source)),
    );
    for (let i = 0; i < parentSets.length; i++) {
      for (let j = i + 1; j < parentSets.length; j++) {
        const overlap = [...parentSets[i]!].some((p) => parentSets[j]!.has(p));
        if (!overlap) return true;
      }
    }
    return trainingNodes.length >= 2;
  }, [nodes, edges]);

  const previewUnavailable = () => previewPending.current
    || useNodeInspectionStore.getState().isLoading || getReadOnlyMode();

  const showValidationResults = () => {
    setExecutionResult(null);
    setLastRunError(null);
    setResultsPanelExpanded(true);
    setResultsPanelDismissed(false);
  };

  const reportPreviewValidation = (issues: ReturnType<typeof collectGraphValidationIssues>) => {
    const panelIssues = issues.filter(issue => issue.category !== 'leakage');
    if (panelIssues.length === 0) {
      notifyPreview('Preview blocked by data leakage. Review the marked nodes on the canvas.');
      return;
    }
    notifyPreview(`Preview blocked. Review ${panelIssues.length} validation issue${panelIssues.length === 1 ? '' : 's'}.`);
    showValidationResults();
  };

  const reportPreviewLeakage = (value: unknown, graphSignature: string): boolean => {
    const leakageMessage = getLeakageErrorMessage(value);
    if (!leakageMessage) return false;
    useNodeInspectionStore.setState({ receipt: null, error: null });
    setLeakageNotice({ message: leakageMessage, graphSignature });
    notifyPreview('Preview blocked by data leakage. Review the canvas safety notice.');
    return true;
  };

  const executePreview = async (graph: ReturnType<typeof useGraphStore.getState>): Promise<void> => {
    previewPending.current = true;
    const graphSignature = graphSemanticSignature(graph.nodes, graph.edges);
    useNotificationsStore.getState().dismiss('canvas-preview');
    setLastRunError(null);
    try {
      const pipelineConfig = buildPreviewConfiguration(graph.nodes, graph.edges);
      const result = await useNodeInspectionStore.getState().runPreview(pipelineConfig);
      if (reportPreviewLeakage(result, graphSignature)) return;
      setExecutionResult(result);
      if (result.status === 'failed') notifyPreview('Preview failed. Open preview results for details.');
      setLastRunError(null);
    } catch (error) {
      if (reportPreviewLeakage(error, graphSignature)) return;
      console.error('Pipeline failed:', error);
      setExecutionResult(null);
      setLastRunError(error instanceof Error ? error.message : String(error));
      setResultsPanelExpanded(true);
      notifyPreview('Preview failed. Open preview results for details.');
    } finally {
      previewPending.current = false;
    }
  };

  const handleRun = async (): Promise<void> => {
    if (previewUnavailable()) return;
    const graph = useGraphStore.getState();
    setLeakageNotice(null);
    const issues = graph.validateGraph();
    if (issues.length > 0) {
      reportPreviewValidation(issues);
      return;
    }
    const datasetNode = graph.nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    if (!datasetId) {
      notifyPreview('Preview blocked. Add a dataset node and select a dataset.');
      return;
    }
    await executePreview(graph);
  };

  const reportExperimentValidation = (issues: ReturnType<typeof collectGraphValidationIssues>) => {
    if (issues.every(issue => issue.category === 'leakage')) {
      notifyExperimentError('Experiments blocked by data leakage. Review the marked nodes on the canvas.');
      return;
    }
    notifyExperimentError('Experiments blocked. Review the validation issues in preview results.');
    showValidationResults();
  };

  const showSubmittedExperiments = (response: RunPipelineResponse) => {
    const count = response.job_ids?.length || 1;
    const jobIds = response.job_ids?.length ? response.job_ids : [response.job_id];
    useNotificationsStore.getState().upsertExecution(`experiments:${jobIds[0]}`,
      `${count} experiment${count > 1 ? 's' : ''} submitted`, { type: 'jobs', run: { label: 'Experiments', jobIds } });
    startPolling();
    if (response.job_ids?.length > 1) {
      setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
    }
    useJobStore.getState().setInspectedRun({ label: 'Experiments', jobIds });
    toggleDrawer(true);
    // Experiments run as background jobs. An inline preview would duplicate
    // that work; users can request canvas data separately with Preview data.
  };

  const reportExperimentFailure = (error: unknown, graphSignature: string) => {
    const leakageMessage = getLeakageErrorMessage(error);
    if (leakageMessage) setLeakageNotice({ message: leakageMessage, graphSignature });
    const message = leakageMessage ?? (error instanceof Error && error.message
      ? error.message : 'Check your connection and settings, then try again.');
    notifyExperimentError(`Experiment submission failed. ${message}`);
  };

  const handleRunAll = async (): Promise<void> => {
    if (experimentsPending.current || previewUnavailable()) return;
    const graph = useGraphStore.getState();
    const { nodes, edges } = graph;
    setLeakageNotice(null);
    const issues = graph.validateGraph();
    if (issues.length > 0) {
      reportExperimentValidation(issues);
      return;
    }
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    if (!datasetId) {
      notifyExperimentError('Experiments blocked. Add a dataset node and select a dataset.');
      return;
    }
    experimentsPending.current = true;
    const graphSignature = graphSemanticSignature(nodes, edges);
    setIsRunningAll(true);
    useNotificationsStore.getState().dismiss('canvas-experiments-error');
    try {
      const previewIds = new Set(
        nodes.filter((n) => n.data.definitionType === 'data_preview').map((n) => n.id),
      );
      const filteredNodes = nodes.filter((n) => !previewIds.has(n.id));
      const filteredEdges = edges.filter(
        (e) => !previewIds.has(e.source) && !previewIds.has(e.target),
      );
      const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
      const response = await jobsApi.runPipeline({ ...pipelineConfig, job_type: 'training' });
      showSubmittedExperiments(response);
    } catch (error) {
      reportExperimentFailure(error, graphSignature);
    } finally {
      setIsRunningAll(false);
      experimentsPending.current = false;
    }
  };

  // Bridge: the global keyboard hook dispatches RUN_PREVIEW_EVENT so we
  // don't have to lift handleRun into a store. The ref always calls the
  // latest closure without re-registering the listener on every render.
  const handleRunRef = useRef<() => void>(() => {});
  handleRunRef.current = () => {
    if (!isRunning) void handleRun();
  };
  useEffect(() => {
    const fire = (): void => handleRunRef.current();
    window.addEventListener(RUN_PREVIEW_EVENT, fire);
    return () => window.removeEventListener(RUN_PREVIEW_EVENT, fire);
  }, []);

  return { isRunning, isRunningAll, canRunPreview, hasMultipleBranches, handleRun, handleRunAll,
    experimentModels, experimentBlockReason };
}
