import { useEffect, useMemo, useRef, useState } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../../../../core/store/useGraphStore';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { runPipelinePreview } from '../../../../core/api/client';
import { jobsApi } from '../../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../../core/utils/pipelineConverter';
import { RUN_PREVIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import { toast } from '../../../../core/toast';

const TRAINING_TYPES = new Set(['training', 'classification', 'regression', 'text_classification']);

export interface RunControls {
  isRunning: boolean;
  isRunningAll: boolean;
  canRunPreview: boolean;
  hasMultipleBranches: boolean;
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

  const validationIssues = useMemo(() => collectGraphValidationIssues(nodes, edges), [nodes, edges]);

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
    const issues = useGraphStore.getState().validateGraph();
    if (issues.length > 0) {
      setExecutionResult(null);
      setLastRunError(null);
      setResultsPanelExpanded(true);
      toast.error(
        'Fix validation issues before running preview',
        'Open Preview Results to inspect the blocking issues.',
      );
      return;
    }

    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    if (!datasetId) {
      toast.error('No dataset node found');
      return;
    }
    setIsRunning(true);
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
        setLastRunError(null);
      } catch (error) {
        console.error('Pipeline failed:', error);
        setExecutionResult(null);
        setLastRunError(error instanceof Error ? error.message : String(error));
        setResultsPanelExpanded(true);
        toast.error('Pipeline execution failed', 'Check console for details.');
      } finally {
        setIsRunning(false);
      }
    };

    const handleRunAll = async (): Promise<void> => {
      const issues = useGraphStore.getState().validateGraph();
      if (issues.length > 0) {
        setExecutionResult(null);
        setLastRunError(null);
        setResultsPanelExpanded(true);
        toast.error(
          'Fix validation issues before running experiments',
          'Open Preview Results to inspect the blocking issues.',
        );
        return;
      }

      const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
      const datasetId = datasetNode?.data.datasetId as string;
      if (!datasetId) {
        toast.error('No dataset node found');
        return;
    }
    setIsRunningAll(true);
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
      if (response.job_ids?.length > 1) {
        setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
        startPolling();
      }
      toast.success(`${count} experiment${count > 1 ? 's' : ''} queued`);
      toggleDrawer();
      // Note: we intentionally do NOT trigger an inline preview here. The
      // experiments run as background jobs; firing a synchronous preview on
      // top would double the work and slow the queue. Users who want live
      // canvas data can click Run Preview separately.
    } catch {
      toast.error('Failed to run experiments');
    } finally {
      setIsRunningAll(false);
    }
  };

  // Bridge: the global keyboard hook dispatches RUN_PREVIEW_EVENT so we
  // don't have to lift handleRun into a store. The ref always calls the
  // latest closure without re-registering the listener on every render.
  const handleRunRef = useRef<() => void>(() => {});
  handleRunRef.current = () => {
    if (!isRunning && canRunPreview) void handleRun();
  };
  useEffect(() => {
    const fire = (): void => handleRunRef.current();
    window.addEventListener(RUN_PREVIEW_EVENT, fire);
    return () => window.removeEventListener(RUN_PREVIEW_EVENT, fire);
  }, []);

  return { isRunning, isRunningAll, canRunPreview, hasMultipleBranches, handleRun, handleRunAll };
}
