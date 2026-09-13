import { useMemo } from 'react';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { upstreamTargetColumns } from '../../../../core/utils/upstreamTargetColumns';

/** Prefer the input's predicted schema so generated coordinates remain selectable. */
export function useGeoDistanceColumns(nodeId?: string) {
  const upstream = useUpstreamData(nodeId || '');
  const datasetId = upstream.find(data => data.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const dropped = useUpstreamDroppedColumns(nodeId);
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const predictions = useGraphStore(state => state.predictedSchemas);
  const targets = useMemo(() => upstreamTargetColumns(nodeId, nodes, edges), [nodeId, nodes, edges]);
  const inputId = edges.find(edge => edge.target === nodeId)?.source;
  const predicted = inputId ? predictions[inputId] : undefined;
  const columns = predicted
    ? predicted.columns.map(name => ({ name, dtype: predicted.dtypes[name] ?? '' }))
    : Object.values(schema?.columns ?? {});
  const numericColumns = columns
    .filter(column => /^(?:u?int\d*|float\d*|double|long|short|number|decimal(?:\([^)]*\))?)$/i.test(column.dtype))
    .filter(column => !dropped.has(column.name) && !targets.has(column.name))
    .map(column => column.name);
  return { datasetId, numericColumns, isLoading: isLoading && !predicted };
}
