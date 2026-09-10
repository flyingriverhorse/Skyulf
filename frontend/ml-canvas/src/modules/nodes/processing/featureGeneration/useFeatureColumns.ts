import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';

export function useFeatureColumns(nodeId?: string) {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  const allColumns = schema ? Object.values(schema.columns).map(c => c.name).filter(n => !droppedUpstream.has(n)) : [];
  const numericColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
        .filter(c => !droppedUpstream.has(c.name))
        .map(c => c.name)
    : [];
  const dateColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['date', 'time'].some(t => c.dtype.toLowerCase().includes(t)))
        .filter(c => !droppedUpstream.has(c.name))
        .map(c => c.name)
    : [];
  const stringColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['string', 'object', 'text'].some(t => c.dtype.toLowerCase().includes(t)))
        .filter(c => !droppedUpstream.has(c.name))
        .map(c => c.name)
    : [];

  return { allColumns, numericColumns, dateColumns, stringColumns };
}
