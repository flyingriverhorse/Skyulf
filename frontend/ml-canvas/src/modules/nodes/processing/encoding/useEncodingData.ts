import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { getNodeMetricDetails } from '../../../../core/utils/preprocessingMetrics';
import type { AnalysisProfile, Recommendation } from '../../../../core/api/client';

function availableColumns(schema: AnalysisProfile | undefined, dropped: Set<string>): string[] {
  return schema ? Object.values(schema.columns).map(column => column.name).filter(name => !dropped.has(name)) : [];
}

function encodingRecommendation(recommendation: Recommendation): boolean {
  return recommendation.suggested_node_type === 'encoding' ||
    recommendation.type.includes('encoding') || recommendation.type.includes('cardinality');
}

export function useEncodingData(nodeId: string | undefined) {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(data => data.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const executionResult = useGraphStore(state => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  return {
    datasetId,
    schema,
    isLoading,
    categoricalColumns: availableColumns(schema, droppedUpstream),
    metrics: getNodeMetricDetails(nodeResult?.metrics),
    filteredRecommendations: (executionResult?.recommendations || []).filter(encodingRecommendation),
  };
}
