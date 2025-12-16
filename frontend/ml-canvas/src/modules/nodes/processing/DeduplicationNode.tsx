import React, { useState, useMemo } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { CopyMinus, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';

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
  const availableColumns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = nodeResult?.metrics;

  const [searchTerm, setSearchTerm] = useState('');

  const filteredColumns = useMemo(() => {
    return availableColumns.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [availableColumns, searchTerm]);

  const handleSelectAll = () => {
    onChange({ ...config, subset: filteredColumns });
  };

  const handleDeselectAll = () => {
    const newSubset = config.subset.filter(c => !filteredColumns.includes(c));
    onChange({ ...config, subset: newSubset });
  };

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
        <label className="block text-sm font-medium mb-1">Keep Strategy</label>
        <select
          className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
          value={config.keep}
          onChange={(e) => onChange({ ...config, keep: e.target.value as any })}
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
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium">Subset Columns (Optional)</label>
          <span className="text-xs text-muted-foreground">
            {config.subset.length} selected
          </span>
        </div>
        <p className="text-xs text-muted-foreground mb-2">
          Only consider these columns for identifying duplicates. If empty, all columns are used.
        </p>

        {/* Search & Actions */}
        <div className="space-y-2 mb-2">
          <input 
            type="text" 
            placeholder="Search columns..." 
            className="block w-full px-3 py-1.5 text-sm bg-background border rounded-md shadow-sm focus:ring-1 focus:ring-primary outline-none"
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); }}
          />
          <div className="flex gap-2 text-xs justify-end">
            <button onClick={handleSelectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Select All</button>
            <span className="text-border">|</span>
            <button onClick={handleDeselectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Deselect All</button>
          </div>
        </div>

        {availableColumns.length > 0 ? (
          <div className="space-y-1 max-h-40 overflow-y-auto border rounded p-2 bg-background">
            {filteredColumns.map(col => (
              <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-1 rounded cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                  checked={config.subset.includes(col)}
                  onChange={(e) => {
                    const newSubset = e.target.checked
                      ? [...config.subset, col]
                      : config.subset.filter(c => c !== col);
                    onChange({ ...config, subset: newSubset });
                  }}
                />
                {col}
              </label>
            ))}
            {filteredColumns.length === 0 && (
              <div className="text-xs text-muted-foreground text-center py-2">
                No columns match "{searchTerm}"
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic border rounded p-4 text-center">
            No columns available
          </div>
        )}
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
              <span className="font-medium text-destructive">{metrics.Deduplicate_rows_removed}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Rows Remaining:</span>
              <span className="font-medium">{metrics.Deduplicate_rows_remaining}</span>
            </div>
            <div className="flex justify-between pt-1 border-t">
              <span className="text-muted-foreground">Total Rows:</span>
              <span className="font-medium">{metrics.Deduplicate_rows_total}</span>
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
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    subset: [],
    keep: 'first',
  }),
};
