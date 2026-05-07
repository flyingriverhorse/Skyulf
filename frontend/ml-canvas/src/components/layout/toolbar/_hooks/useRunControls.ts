import { useEffect, useMemo, useRef, useState } from 'react';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { useJobStore } from '../../../../core/store/useJobStore';
import { runPipelinePreview } from '../../../../core/api/client';
import { jobsApi } from '../../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../../core/utils/pipelineConverter';
import { RUN_PREVIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import { toast } from '../../../../core/toast';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);

export interface RunControls {
  isRunning: boolean;
  isRunningAll: boolean;
  canRunPreview: boolean;
  hasMultipleBranches: boolean;
  handleRun: () => Promise<void>;
  handleRunAll: () => Promise<void>;
}

export function useRunControls(): RunControls {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const setExecutionResult = useGraphStore((s) => s.setExecutionResult);
  const { toggleDrawer, setActiveParallelRun, startPolling } = useJobStore();

  const [isRunning, setIsRunning] = useState(false);
  const [isRunningAll, setIsRunningAll] = useState(false);

  const canRunPreview = useMemo(() => {
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    if (!datasetNode) return false;
    const datasetId = datasetNode.data.datasetId as string | undefined;
    if (!datasetId) return false;
    return edges.some((e) => e.source === datasetNode.id);
  }, [nodes, edges]);

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
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    if (!datasetId) {
      toast.error('No dataset node found');
      return;
    }
    setIsRunning(true);
    setExecutionResult(null);
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
    } catch (error) {
      console.error('Pipeline failed:', error);
      toast.error('Pipeline execution failed', 'Check console for details.');
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunAll = async (): Promise<void> => {
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
        job_type: 'basic_training',
      });
      const count = response.job_ids?.length || 1;
      if (response.job_ids?.length > 1) {
        setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
        startPolling();
      }
      toast.success(`${count} experiment${count > 1 ? 's' : ''} queued`);
      toggleDrawer();
      // Keep standard nodes populated with data while experiments run.
      void handleRun();
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
