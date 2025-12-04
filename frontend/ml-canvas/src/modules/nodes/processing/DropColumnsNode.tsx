import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Trash2 } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface DropColumnsConfig {
  columns: string[];
}

const DropColumnsSettings: React.FC<{ config: DropColumnsConfig; onChange: (c: DropColumnsConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const availableColumns = schema?.columns.map(c => c.name) || [];

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
        <label className="block text-sm font-medium mb-2">Select Columns to Drop</label>
        {availableColumns.length > 0 ? (
          <div className="space-y-1 max-h-60 overflow-y-auto border rounded p-2 bg-background">
            {availableColumns.map(col => (
              <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-1 rounded cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.columns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.columns, col]
                      : config.columns.filter(c => c !== col);
                    onChange({ ...config, columns: newCols });
                  }}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span>{col}</span>
              </label>
            ))}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic border rounded p-4 text-center">
            No columns available
          </div>
        )}
        <p className="text-[10px] text-muted-foreground mt-2">
          Selected columns will be removed from the dataset.
        </p>
      </div>
    </div>
  );
};

export const DropColumnsNode: NodeDefinition<DropColumnsConfig> = {
  type: 'drop_missing_columns', // Maps to backend handler
  label: 'Drop Columns',
  category: 'Preprocessing',
  description: 'Remove specific columns from the dataset.',
  icon: Trash2,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Data', type: 'dataset' }],
  settings: DropColumnsSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0,
    message: config.columns.length === 0 ? 'Select at least one column to drop' : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
  }),
};
