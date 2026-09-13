import { useEffect, useMemo, useState } from 'react';
import { apiClient } from '../api/client';
import { useGraphStore } from '../store/useGraphStore';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import type { Node, Edge } from '@xyflow/react';

interface PruningContext {
  enabled: boolean;
  nodeId?: string | undefined;
  modelType?: string | undefined;
  searchSpace?: Record<string, unknown> | undefined;
  strategyParams?: Record<string, unknown> | undefined;
}

export interface PruningSupport {
  supported?: boolean;
  mode?: 'iterations' | 'folds' | 'none';
  reason: string;
}

/** Explain the stopping boundary when no more specific backend reason is supplied. */
function supportReason(mode: PruningSupport['mode']): string {
  if (mode === 'folds') return 'Early stopping is available between CV folds.';
  if (mode === 'iterations') return 'Early stopping is available during training.';
  return 'Early stopping is unavailable for this model and pipeline.';
}

/** Use executable graph/configuration changes, ignoring pointer positions and selection. */
function requestKey(context: PruningContext, nodes: Node[], edges: Edge[]): string | null {
  if (!context.enabled || !context.nodeId || !context.modelType) return null;
  const pipeline = convertGraphToPipelineConfig(nodes, edges);
  if (!pipeline.nodes.some(node => node.node_id === context.nodeId)) return null;
  return JSON.stringify({
    node_id: context.nodeId,
    model_type: context.modelType,
    search_space: context.searchSpace ?? {},
    strategy_params: context.strategyParams ?? {},
    pipeline: { ...pipeline, pipeline_id: 'pruning-support' },
  });
}

/** Ask the training backend for capability without writing defaults into graph history. */
export function useOptunaPruningSupport(context: PruningContext): PruningSupport {
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const { enabled, nodeId, modelType, searchSpace, strategyParams } = context;
  const key = useMemo(() => requestKey({ enabled, nodeId, modelType, searchSpace, strategyParams }, nodes, edges),
    [enabled, nodeId, modelType, searchSpace, strategyParams, nodes, edges]);
  const [result, setResult] = useState<{ key: string; support: PruningSupport } | null>(null);

  useEffect(() => {
    setResult(null);
    if (key === null) return;
    const controller = new AbortController();
    let active = true;
    apiClient.post<{ supported: boolean; mode: NonNullable<PruningSupport['mode']>; reason: string | null }>(
      '/pipeline/pruning-support', JSON.parse(key), { signal: controller.signal },
    ).then(({ data }) => {
      if (active) setResult({ key, support: {
        supported: data.supported,
        mode: data.mode,
        reason: data.reason ?? supportReason(data.mode),
      } });
    }).catch(() => {
      if (active) setResult({ key, support: {
        reason: 'Could not check pruning support. Reopen these settings to retry; your saved choice is unchanged.',
      } });
    });
    return () => { active = false; controller.abort(); };
  }, [key]);

  if (key === null) return { reason: 'Connect this model to a dataset to check pruning support.' };
  if (result?.key === key) return result.support;
  return { reason: 'Checking pruning support for this model and pipeline…' };
}
