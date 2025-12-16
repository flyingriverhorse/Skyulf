import React, { useState, useMemo } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Flag, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface MissingIndicatorConfig {
  columns: string[];
  flag_suffix: string;
}

const MissingIndicatorSettings: React.FC<{ config: MissingIndicatorConfig; onChange: (c: MissingIndicatorConfig) => void; nodeId?: string }> = ({
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
    onChange({ ...config, columns: filteredColumns });
  };

  const handleDeselectAll = () => {
    const newCols = config.columns.filter(c => !filteredColumns.includes(c));
    onChange({ ...config, columns: newCols });
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
        <label className="block text-sm font-medium mb-1">Indicator Suffix</label>
        <input
          type="text"
          className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
          value={config.flag_suffix}
          onChange={(e) => onChange({ ...config, flag_suffix: e.target.value })}
          placeholder="_was_missing"
        />
        <p className="text-xs text-muted-foreground mt-1">
          Suffix added to the new indicator columns (e.g., "Age_was_missing").
        </p>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium">Target Columns (Optional)</label>
          <span className="text-xs text-muted-foreground">
            {config.columns.length} selected
          </span>
        </div>
        <p className="text-xs text-muted-foreground mb-2">
          Create indicators for these columns. If empty, all columns with missing values are used.
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
                  checked={config.columns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.columns, col]
                      : config.columns.filter(c => c !== col);
                    onChange({ ...config, columns: newCols });
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
      {metrics && (metrics.missing_indicators_created !== undefined) && (
        <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
            <Activity size={14} />
            <span>Last Run Results</span>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Indicators Created:</span>
              <span className="font-medium text-primary">{metrics.missing_indicators_created}</span>
            </div>
            {metrics.missing_indicators_columns && Array.isArray(metrics.missing_indicators_columns) && metrics.missing_indicators_columns.length > 0 && (
              <div className="pt-1 border-t mt-1">
                <span className="text-muted-foreground block mb-1">New Columns:</span>
                <div className="flex flex-wrap gap-1">
                  {metrics.missing_indicators_columns.map((col: string) => (
                    <span key={col} className="px-1.5 py-0.5 bg-background border rounded text-[10px] font-mono">
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const MissingIndicatorNode: NodeDefinition<MissingIndicatorConfig> = {
  type: 'MissingIndicator',
  label: 'Missing Indicator',
  category: 'Preprocessing',
  description: 'Create binary indicators for missing values.',
  icon: Flag,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Augmented Data', type: 'dataset' }],
  settings: MissingIndicatorSettings,
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    columns: [],
    flag_suffix: '_was_missing',
  }),
};
