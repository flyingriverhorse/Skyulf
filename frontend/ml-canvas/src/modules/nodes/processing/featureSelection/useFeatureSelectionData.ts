import { useEffect } from 'react';
import { getIncomers, type Node, type Edge } from '@xyflow/react';
import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import type { FeatureSelectionConfig, SelectionConfigProps } from './types';

function findUpstreamDatasetId(currentNodeId: string, nodes: Node[], edges: Edge[]): string | undefined {
  const visited = new Set<string>();
  const queue = [currentNodeId];

  while (queue.length > 0) {
    const id = queue.shift();
    if (!id) continue;
    if (visited.has(id)) continue;
    visited.add(id);

    const node = nodes.find(n => n.id === id);
    if (!node) continue;

    // If this is NOT the current node, check if it has datasetId
    if (id !== currentNodeId && node.data?.datasetId) {
      return node.data.datasetId as string;
    }

    const incomers = getIncomers(node, nodes, edges);
    for (const incomer of incomers) {
      queue.push(incomer.id);
    }
  }
  return undefined;
}

export function useFeatureSelectionData({ config, onChange, nodeId }: SelectionConfigProps & { nodeId?: string | undefined }) {
  const upstreamData = useUpstreamData(nodeId || '');

  // Recursive search for datasetId
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const executionResult = useGraphStore((state) => state.executionResult);

  const upstreamDatasetId = findUpstreamDatasetId(nodeId || '', nodes, edges);
  const upstreamTargetColumn = upstreamData.find((d: Record<string, unknown>) => d.target_column)?.target_column as string | undefined;

  // Auto-detect target and dataset
  useEffect(() => {
    const updates: Partial<FeatureSelectionConfig> = {};
    // Propagate datasetId, and clear it if the upstream dataset connection is removed
    // (otherwise a stale datasetId lingers on this node's data forever, making
    // downstream nodes think a dataset is still connected when it isn't).
    if (upstreamDatasetId && config.datasetId !== upstreamDatasetId) {
      updates.datasetId = upstreamDatasetId;
    } else if (!upstreamDatasetId && config.datasetId) {
      updates.datasetId = undefined;
    }
    if (upstreamTargetColumn && config.target_column !== upstreamTargetColumn) {
      updates.target_column = upstreamTargetColumn;
    }
    if (Object.keys(updates).length > 0) {
      onChange({ ...config, ...updates });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamDatasetId, upstreamTargetColumn, config.datasetId, config.target_column, onChange]);

  const { data: schema, isLoading } = useDatasetSchema(upstreamDatasetId || config.datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name).filter(n => !droppedUpstream.has(n)) : [];


  return { upstreamDatasetId, upstreamTargetColumn, columns, isLoading, result: executionResult?.node_results[nodeId || ''] };
}
