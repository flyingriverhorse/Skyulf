import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { getNodeMetricDetails } from '../../../../core/utils/preprocessingMetrics';

export function useScalingData(nodeId?: string) {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = getNodeMetricDetails(nodeResult?.metrics);

  // Filter for numeric columns only, as scaling only applies to them
  const numericColumns = schema
    ? Object.values(schema.columns)
      .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
      .filter(c => !droppedUpstream.has(c.name))
      .map(c => c.name)
    : [];

  return { datasetId, isLoading, numericColumns, metrics };
}
