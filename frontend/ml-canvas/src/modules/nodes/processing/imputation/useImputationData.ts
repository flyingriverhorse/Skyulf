import { useMemo } from 'react';
import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { useRecommendations } from '../../../../core/hooks/useRecommendations';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { getNodeMetricDetails } from '../../../../core/utils/preprocessingMetrics';

/** Preserve object-valued metric details, including empty maps and array payloads. */
function feedbackMappings(metrics: Record<string, unknown> | null) {
  const fillValues: Record<string, unknown> | null =
    metrics?.fill_values && typeof metrics.fill_values === 'object'
      ? (metrics.fill_values as Record<string, unknown>)
      : null;
  const missingCounts: Record<string, unknown> | null =
    metrics?.missing_counts && typeof metrics.missing_counts === 'object'
      ? (metrics.missing_counts as Record<string, unknown>)
      : null;

  return { fillValues, missingCounts };
}

/** Resolve the first upstream dataset and shared schema, recommendation and feedback state. */
export function useImputationData(nodeId: string | undefined) {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const availableColumns = useMemo(
    () => schema ? Object.values(schema.columns).map(c => c.name).filter(n => !droppedUpstream.has(n)) : [],
    [schema, droppedUpstream]
  );

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = getNodeMetricDetails(nodeResult?.metrics);
  const recommendations = useRecommendations(nodeId || '', {
    types: ['imputation'],
    suggestedNodeTypes: ['SimpleImputer', 'KNNImputer', 'IterativeImputer'],
    scope: 'column'
  });

  return { datasetId, isLoading, availableColumns, nodeResult, metrics, recommendations, ...feedbackMappings(metrics) };
}

export type ImputationData = ReturnType<typeof useImputationData>;
