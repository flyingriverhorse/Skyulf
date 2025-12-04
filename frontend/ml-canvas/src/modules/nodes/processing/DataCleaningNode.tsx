import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Eraser } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface DataCleaningConfig {
  dropColumns: string[];
  fillStrategy: 'mean' | 'median' | 'mode' | 'drop';
}

const DataCleaningSettings: React.FC<{ config: DataCleaningConfig; onChange: (c: DataCleaningConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  
  // Find a datasetId in upstream nodes
  // We look for any upstream node that has a datasetId property
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const columns = schema?.columns.map(c => c.name) || [];

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
        <label className="block text-sm font-medium mb-1">Fill Strategy</label>
        <select
          className="w-full p-2 border rounded bg-background"
          value={config.fillStrategy}
          onChange={(e) => onChange({ ...config, fillStrategy: e.target.value as any })}
        >
          <option value="mean">Mean</option>
          <option value="median">Median</option>
          <option value="mode">Mode</option>
          <option value="drop">Drop Rows</option>
        </select>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-1">Drop Columns</label>
        {columns.length > 0 ? (
          <div className="space-y-1 max-h-40 overflow-y-auto border rounded p-2">
            {columns.map(col => (
              <label key={col} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.dropColumns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.dropColumns, col]
                      : config.dropColumns.filter(c => c !== col);
                    onChange({ ...config, dropColumns: newCols });
                  }}
                />
                {col}
              </label>
            ))}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic border rounded p-4 text-center">
            No columns available
          </div>
        )}
      </div>
    </div>
  );
};

export const DataCleaningNode: NodeDefinition<DataCleaningConfig> = {
  type: 'data_cleaning',
  label: 'Data Cleaning',
  category: 'Preprocessing',
  description: 'Clean and preprocess data.',
  icon: Eraser,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: DataCleaningSettings,
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    dropColumns: [],
    fillStrategy: 'mean',
  }),
};
