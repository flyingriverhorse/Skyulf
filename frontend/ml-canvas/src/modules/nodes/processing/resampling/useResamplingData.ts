import { useEffect } from 'react';
import type { ColumnProfile } from '../../../../core/api/client';
import { useUpstreamData } from '../../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../../core/hooks/useUpstreamDroppedColumns';
import type { ResamplingNode } from '../ResamplingNode';

type Config = ReturnType<typeof ResamplingNode.getDefaultConfig>;

/** Resolve the ancestor schema and target suggestion without changing autofill timing. */
export function useResamplingData(nodeId: string | undefined, config: Config, onChange: (next: Config) => void) {
  // Upstream Data for Target Column Suggestion
  const upstreamDataRaw = useUpstreamData(nodeId || '') as unknown;
  const upstreamData: Record<string, unknown>[] = Array.isArray(upstreamDataRaw)
    ? (upstreamDataRaw.filter(Boolean) as Record<string, unknown>[])
    : [];

  const datasetId = upstreamData.find((d) => typeof d.datasetId === 'string')?.datasetId as
    | string
    | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  // Try to find a target column from upstream nodes configuration
  const upstreamTarget = upstreamData.find((d) => {
    const cfg = d.config;
    if (!cfg || typeof cfg !== 'object') return false;
    const target = (cfg as Record<string, unknown>).target_column;
    return typeof target === 'string' && target.trim().length > 0;
  });
  const targetColumn = upstreamTarget
    ? (((upstreamTarget.config as Record<string, unknown>).target_column as string) ?? undefined)
    : undefined;

  // Auto-fill target column if empty and available in schema or upstream
  useEffect(() => {
    if (!config.target_column) {
      if (upstreamTarget && targetColumn) {
        onChange({ ...config, target_column: targetColumn });
      } else if (schema?.columns) {
        // Simple heuristic: check for 'target' or 'class' or column_type
        const potentialTarget = Object.values(schema.columns).find((c: ColumnProfile) =>
          c.name.toLowerCase() === 'target' ||
          c.name.toLowerCase() === 'class' ||
          c.column_type === 'target'
        );
        if (potentialTarget) {
          onChange({ ...config, target_column: potentialTarget.name });
        }
      }
    }
  }, [schema, upstreamTarget, targetColumn, config.target_column, config, onChange]);

  return { schema, droppedUpstream, upstreamTarget };
}
