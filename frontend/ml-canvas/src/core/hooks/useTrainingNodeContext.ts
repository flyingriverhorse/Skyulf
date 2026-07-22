import { useCallback, useMemo } from 'react';
import { getIncomers, type Edge, type Node } from '@xyflow/react';
import { useGraphStore } from '../store/useGraphStore';
import { useJobStore } from '../store/useJobStore';
import { useUpstreamData } from './useUpstreamData';
import { useDatasetSchema } from './useDatasetSchema';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import { warnAndBlockOnLeakage } from '../utils/pipelineLeakageValidation';
import { jobsApi } from '../api/jobs';
import { toast } from '../toast';
import type { TaskType } from '../types/taskType';

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
      if (!nodeId) return;
      try {
        const cfg = convertGraphToPipelineConfig(nodes, edges);
        if (warnAndBlockOnLeakage(cfg)) return;
        const res = await jobsApi.runPipeline({
          ...cfg,
          target_node_id: nodeId,
          job_type: jobType,
        });
        const count = res.job_ids?.length ?? 1;
        if (count > 1) {
          setActiveParallelRun({ jobIds: res.job_ids, startedAt: new Date().toISOString() });
          startPolling();
          toast.success('Parallel execution started', `${count} branches submitted.`);
        } else {
          toast.success('Training job submitted');
        }
        setTab(task);
        toggleDrawer(true);
      } catch (error) {
        console.error('Failed to submit job:', error);
        toast.error('Failed to submit job');
      }
    },
    [nodeId, nodes, edges, setActiveParallelRun, startPolling, setTab, toggleDrawer],
  );

  return { availableColumns, upstreamTarget, datasetId, runJob };
}
