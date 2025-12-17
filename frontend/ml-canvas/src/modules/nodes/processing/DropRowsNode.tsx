import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { FilterX, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface DropRowsConfig {
  drop_if_any_missing: boolean;
  missing_threshold?: number; // 0-100
}

const DropRowsSettings: React.FC<{ config: DropRowsConfig; onChange: (c: DropRowsConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { isLoading } = useDatasetSchema(datasetId);
  
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

      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm font-medium">
          <input
            type="checkbox"
            checked={config.drop_if_any_missing}
            onChange={(e) => onChange({ ...config, drop_if_any_missing: e.target.checked })}
            className="rounded border-gray-300"
          />
          Drop rows with ANY missing values
        </label>
        <p className="text-xs text-muted-foreground">
          If checked, any row containing at least one missing value will be removed.
        </p>
      </div>

      <div className={`space-y-2 ${config.drop_if_any_missing ? 'opacity-50 pointer-events-none' : ''}`}>
        <label className="block text-sm font-medium">
          Missing Value Threshold (%)
        </label>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min="0"
            max="100"
            value={config.missing_threshold || 0}
            onChange={(e) => onChange({ ...config, missing_threshold: parseInt(e.target.value) })}
            className="flex-1"
          />
          <span className="text-sm w-12 text-right">{config.missing_threshold || 0}%</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Drop rows that have more than this percentage of missing values.
        </p>
      </div>

      {/* Feedback Section */}
      {metrics && (metrics.DropMissingRows_rows_removed !== undefined) && (
        <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
            <Activity size={14} />
            <span>Last Run Results</span>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rows Removed:</span>
              <span className="font-medium text-destructive">{String(metrics.DropMissingRows_rows_removed)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rows Remaining:</span>
              <span className="font-medium">{String(metrics.DropMissingRows_rows_remaining)}</span>
            </div>
            <div className="flex justify-between pt-1 border-t">
              <span className="text-muted-foreground">Total Rows:</span>
              <span className="font-medium">{String(metrics.DropMissingRows_rows_total)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const DropRowsNode: NodeDefinition<DropRowsConfig> = {
  type: 'drop_missing_rows',
  label: 'Drop Rows',
  category: 'Preprocessing',
  description: 'Drop rows based on missing values.',
  icon: FilterX,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: DropRowsSettings,
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    drop_if_any_missing: false,
    missing_threshold: 50,
  }),
};
