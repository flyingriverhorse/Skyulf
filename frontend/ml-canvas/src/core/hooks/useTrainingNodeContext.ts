import { useCallback, useMemo } from 'react';
import { getIncomers, type Edge, type Node } from '@xyflow/react';
import { useGraphStore } from '../store/useGraphStore';
import { useJobStore } from '../store/useJobStore';
import { useViewStore } from '../store/useViewStore';
import { useUpstreamData } from './useUpstreamData';
import { useDatasetSchema } from './useDatasetSchema';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import { warnAndBlockOnLeakage } from '../utils/pipelineLeakageValidation';
import { getLeakageErrorMessage, graphSemanticSignature } from '../utils/leakageFeedback';
import { jobsApi } from '../api/jobs';
import { toast } from '../toast';
import type { TaskType } from '../types/taskType';
import type { NodeSubmission } from '../types/runFeedback';

type JobType = 'training' | 'tuning';

/**
 * Walk upstream (multi-hop) to find the dataset id feeding a training-style
 * node. Mirrors the resolver embedded in the Basic/Advanced training panels so
 * new model nodes don't have to re-implement it.
 */
function findUpstreamDatasetId(
  nodeId: string | undefined,
  nodes: Node[],
  edges: Edge[],
): string | undefined {
  if (!nodeId) return undefined;
  const visited = new Set<string>();
  const queue = [nodeId];
  while (queue.length > 0) {
    const id = queue.shift();
    if (!id || visited.has(id)) continue;
    visited.add(id);
    const node = nodes.find((n) => n.id === id);
    if (!node) continue;

    if (id !== nodeId) {
      const data = node.data as Record<string, unknown> | undefined;
      const fromData = (data?.datasetId ?? data?.dataset_id) as string | undefined;
      if (fromData) return fromData;
      const cfg = data?.config as Record<string, unknown> | undefined;
      const fromCfg = (cfg?.datasetId ?? cfg?.dataset_id) as string | undefined;
      if (fromCfg) return fromCfg;
      const params = data?.params as Record<string, unknown> | undefined;
      const fromParams = (params?.datasetId ?? params?.dataset_id) as string | undefined;
      if (fromParams) return fromParams;
    }

    for (const inc of getIncomers(node, nodes, edges)) queue.push(inc.id);
  }
  return undefined;
}

/**
 * Shared context for model-training nodes: the upstream dataset schema columns,
 * the auto-detected target column, and a `runJob` action that submits the
 * pipeline. Extracted so new training-style nodes (e.g. the Ensemble node) reuse
 * the same plumbing instead of duplicating it.
 */
export function useTrainingNodeContext(nodeId: string | undefined) {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const upstreamData = useUpstreamData(nodeId || '');
  const { toggleDrawer, setTab, setActiveParallelRun, startPolling } = useJobStore();
  const feedback = useJobStore(state => nodeId ? state.nodeSubmissions[nodeId] : undefined);

  const datasetId = useMemo(
    () => findUpstreamDatasetId(nodeId, nodes, edges),
    [nodeId, nodes, edges],
  );
  const { data: schema } = useDatasetSchema(datasetId);
  const availableColumns = useMemo(
    () => (schema ? Object.values(schema.columns) : []),
    [schema],
  );
  const upstreamTarget = upstreamData.find((d) => d.target_column)?.target_column as
    | string
    | undefined;

  const runJob = useCallback(
    async (jobType: JobType, task: TaskType) => {
      if (!nodeId || useJobStore.getState().nodeSubmissions[nodeId]?.pending) return;
      const node = nodes.find(item => item.id === nodeId);
      if (!node) return;
      const modelName = String(node.data.model_type || node.data.label || 'selected model').replace(/_/g, ' ');
      const label = `${jobType === 'tuning' ? 'Tuning' : 'Training'} — ${modelName}`;
      const update = (value: NodeSubmission) => useJobStore.getState().setNodeSubmission(nodeId, value);
      if (!datasetId) {
        update({ pending: false, run: null, message: `${label} blocked. Connect a dataset upstream and select a dataset.` });
        return;
      }
      useViewStore.getState().setLeakageNotice(null);
      const graphSignature = graphSemanticSignature(nodes, edges);
      update({ pending: true, run: null, message: `${label}: Submitting...` });
      try {
        const cfg = convertGraphToPipelineConfig(nodes, edges);
        // Match backend target scoping while retaining the full graph for submission.
        const nodesById = new Map(cfg.nodes.map(configNode => [configNode.node_id, configNode]));
        const selectedNodeIds = new Set<string>();
        const pendingNodeIds = [nodeId];
        while (pendingNodeIds.length > 0) {
          const currentId = pendingNodeIds.pop()!;
          if (selectedNodeIds.has(currentId)) continue;
          selectedNodeIds.add(currentId);
          pendingNodeIds.push(...(nodesById.get(currentId)?.inputs ?? []));
        }
        const selectedNodes = cfg.nodes.filter(configNode => selectedNodeIds.has(configNode.node_id));
        if (warnAndBlockOnLeakage({ nodes: selectedNodes })) {
          update({ pending: false, run: null, message: `${label} blocked. Move data-learning preprocessing after the train/test split.` });
          return;
        }
        const res = await jobsApi.runPipeline({
          ...cfg,
          target_node_id: nodeId,
          job_type: jobType,
        });
        const jobIds = res.job_ids?.length ? res.job_ids : [res.job_id];
        const count = jobIds.length;
        update({ pending: false, message: '', run: { label, jobIds } });
        startPolling();
        if (count > 1) {
          setActiveParallelRun({ jobIds: res.job_ids, startedAt: new Date().toISOString() });
          toast.success('Parallel execution started', `${count} branches submitted.`);
        } else {
          toast.success(`${jobType === 'tuning' ? 'Tuning' : 'Training'} job submitted`);
        }
        setTab(task);
        useJobStore.getState().setInspectedRun(null);
        toggleDrawer(true);
      } catch (error) {
        console.error('Failed to submit job:', error);
        const leakageMessage = getLeakageErrorMessage(error);
        if (leakageMessage) useViewStore.getState().setLeakageNotice({ message: leakageMessage, graphSignature });
        const message = leakageMessage ?? (error instanceof Error && error.message
          ? error.message : 'Check your connection and settings, then try again.');
        update({ pending: false, run: null, message: `${label}: Submission failed. ${message}` });
        toast.error('Failed to submit job', message);
      }
    },
    [nodeId, nodes, edges, datasetId, setActiveParallelRun, startPolling, setTab, toggleDrawer],
  );

  return { availableColumns, upstreamTarget, datasetId, runJob,
    isSubmitting: feedback?.pending ?? false,
    submissionMessage: feedback?.message ?? '',
    runFeedback: feedback?.run ?? null,
  };
}
