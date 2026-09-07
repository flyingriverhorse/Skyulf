import { useEffect, useMemo, useRef, useState } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../../../../core/store/useGraphStore';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { useNotificationsStore } from '../../../../core/store/useNotificationsStore';
import { runPipelinePreview } from '../../../../core/api/client';
import { jobsApi } from '../../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../../core/utils/pipelineConverter';
import { RUN_PREVIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import { registry } from '../../../../core/registry/NodeRegistry';
import { nodeDisplayNames } from '../../../../core/utils/nodeDisplayNames';
import { getReadOnlyMode } from '../../../../core/hooks/useReadOnlyMode';

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
  const { setResultsPanelExpanded } = useViewStore();

  const [isRunning, setIsRunning] = useState(false);
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

  const handleRun = async (): Promise<void> => {
    if (previewPending.current || getReadOnlyMode()) return;
    const issues = useGraphStore.getState().validateGraph();
    if (issues.length > 0) {
      notifyPreview(`Preview blocked. Review ${issues.length} validation issue${issues.length === 1 ? '' : 's'}.`);
      setExecutionResult(null);
      setLastRunError(null);
      setResultsPanelExpanded(true);
      return;
    }

    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    if (!datasetId) {
      notifyPreview('Preview blocked. Add a dataset node and select a dataset.');
      return;
    }
    previewPending.current = true;
    setIsRunning(true);
    useNotificationsStore.getState().dismiss('canvas-preview');
    setExecutionResult(null);
    setLastRunError(null);
      try {
        // Exclude Data Preview nodes — they're inspection sinks, not pipeline steps.
        const previewIds = new Set(
          nodes.filter((n) => n.data.definitionType === 'data_preview').map((n) => n.id),
        );
        const filteredNodes = nodes.filter((n) => !previewIds.has(n.id));
        const filteredEdges = edges.filter(
          (e) => !previewIds.has(e.source) && !previewIds.has(e.target),
        );
        const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
      const result = await runPipelinePreview(pipelineConfig);
      setExecutionResult(result);
      if (result.status === 'failed') notifyPreview('Preview failed. Open preview results for details.');
        setLastRunError(null);
      } catch (error) {
        console.error('Pipeline failed:', error);
        setExecutionResult(null);
        setLastRunError(error instanceof Error ? error.message : String(error));
        setResultsPanelExpanded(true);
        notifyPreview('Preview failed. Open preview results for details.');
      } finally {
        setIsRunning(false);
        previewPending.current = false;
      }
    };

    const handleRunAll = async (): Promise<void> => {
      if (experimentsPending.current || previewPending.current || getReadOnlyMode()) return;
      const issues = useGraphStore.getState().validateGraph();
      if (issues.length > 0) {
        notifyExperimentError('Experiments blocked. Review the validation issues in preview results.');
        setExecutionResult(null);
        setLastRunError(null);
        setResultsPanelExpanded(true);
        return;
      }

      const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
      const datasetId = datasetNode?.data.datasetId as string;
      if (!datasetId) {
        notifyExperimentError('Experiments blocked. Add a dataset node and select a dataset.');
        return;
    }
    experimentsPending.current = true;
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
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'training',
      });
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
      // Note: we intentionally do NOT trigger an inline preview here. The
      // experiments run as background jobs; firing a synchronous preview on
      // top would double the work and slow the queue. Users who want live
      // canvas data can click Run Preview separately.
    } catch {
      notifyExperimentError('Experiment submission failed. Check your connection and settings, then try again.');
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
