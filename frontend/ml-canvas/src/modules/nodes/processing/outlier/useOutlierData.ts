import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { getIncomers } from '@xyflow/react';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { useRecommendations } from '../../../../core/hooks/useRecommendations';
import { getNodeMetricDetails } from '../../../../core/utils/preprocessingMetrics';

export function useOutlierData(nodeId?: string) {
  // Recursive search for datasetId
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
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
  };

  const datasetId = findUpstreamDatasetId(nodeId || '');
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  const backendRecommendations = useRecommendations(nodeId || '', {
    types: ['outlier_removal', 'cleaning'],
    suggestedNodeTypes: ['OutlierRemoval', 'outlier'],
    scope: 'column'
  });

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = getNodeMetricDetails(nodeResult?.metrics);

  // Filter for numeric columns only
  const numericColumns = schema
    ? Object.values(schema.columns)
      .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
      .filter(c => !droppedUpstream.has(c.name))
      .map(c => c.name)
    : [];

  return { datasetId, isLoading, numericColumns, metrics, nodeResult, backendRecommendations };
}
