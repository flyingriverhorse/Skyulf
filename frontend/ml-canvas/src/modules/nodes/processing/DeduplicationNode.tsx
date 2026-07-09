import React, { useMemo } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { CopyMinus, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';

interface DeduplicationConfig {
  subset: string[];
  keep: 'first' | 'last' | 'none';
}

const DeduplicationSettings: React.FC<{ config: DeduplicationConfig; onChange: (c: DeduplicationConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
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
  const metrics: Record<string, unknown> | null =
    nodeResult?.metrics && typeof nodeResult.metrics === 'object'
      ? (nodeResult.metrics as Record<string, unknown>)
      : null;

  return (
    <div className="p-4 space-y-4">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
          Connect a dataset node to see available columns.
        </div>
      )}

      {isLoading && !!datasetId && (
        <div className="text-xs text-muted-foreground animate-pulse">
          Loading schema...
        </div>
      )}

      <div>
        <span className="block text-sm font-medium mb-1">Keep Strategy</span>
        <select
          className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
          value={config.keep}
          onChange={(e) => onChange({ ...config, keep: e.target.value as DeduplicationConfig['keep'] })}
        >
          <option value="first">Keep First Occurrence</option>
          <option value="last">Keep Last Occurrence</option>
          <option value="none">Drop All Duplicates</option>
        </select>
        <p className="text-xs text-muted-foreground mt-1">
          Determines which duplicates (if any) to keep.
        </p>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-2">
          Only consider these columns for identifying duplicates. If empty, all columns are used.
        </p>

        <ColumnMultiSelect
          columns={availableColumns}
          selected={config.subset}
          onChange={(newSubset) => { onChange({ ...config, subset: newSubset }); }}
          label="Subset Columns (Optional)"
          variant="panel"
          isLoading={isLoading}
          fillHeight={false}
        />
      </div>

      {/* Feedback Section */}
      {metrics && (metrics.Deduplicate_rows_removed !== undefined) && (
        <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
            <Activity size={14} />
            <span>Last Run Results</span>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Duplicates Removed:</span>
              <span className="font-medium text-destructive">{String(metrics.Deduplicate_rows_removed)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rows Remaining:</span>
              <span className="font-medium">{String(metrics.Deduplicate_rows_remaining)}</span>
            </div>
            <div className="flex justify-between pt-1 border-t">
              <span className="text-muted-foreground">Total Rows:</span>
              <span className="font-medium">{String(metrics.Deduplicate_rows_total)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const DeduplicationNode: NodeDefinition<DeduplicationConfig> = {
  type: 'deduplicate',
  label: 'Deduplicate',
  category: 'Preprocessing',
  description: 'Remove duplicate rows.',
  icon: CopyMinus,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Unique Data', type: 'dataset' }],
  settings: DeduplicationSettings,
  bodyPreview: (config) => {
    const subset = config.subset?.length ?? 0;
    const keep = config.keep ?? 'first';
    const subsetStr = subset === 0 ? 'all' : `${subset} ${subset === 1 ? 'col' : 'cols'}`;
    return `Subset: ${subsetStr} · keep ${keep}`;
  },
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    subset: [],
    keep: 'first',
  }),
};
