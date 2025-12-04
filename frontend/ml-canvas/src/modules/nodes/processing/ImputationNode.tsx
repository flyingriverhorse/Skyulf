import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { PaintBucket } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface ImputationConfig {
  columns: string[];
  strategy: 'mean' | 'median' | 'most_frequent' | 'constant';
  fill_value?: string;
}

const ImputationSettings: React.FC<{ config: ImputationConfig; onChange: (c: ImputationConfig) => void; nodeId?: string }> = ({
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
        <label className="block text-sm font-medium mb-1">Imputation Strategy</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.strategy}
          onChange={(e) => onChange({ ...config, strategy: e.target.value as any })}
        >
          <option value="mean">Mean (Average)</option>
          <option value="median">Median (Middle Value)</option>
          <option value="most_frequent">Most Frequent (Mode)</option>
          <option value="constant">Constant Value</option>
        </select>
      </div>

      {config.strategy === 'constant' && (
        <div>
          <label className="block text-sm font-medium mb-1">Fill Value</label>
          <input
            type="text"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.fill_value || ''}
            onChange={(e) => onChange({ ...config, fill_value: e.target.value })}
            placeholder="Enter value..."
          />
        </div>
      )}
      
      <div>
        <label className="block text-sm font-medium mb-2">Target Columns</label>
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
      </div>
    </div>
  );
};

export const ImputationNode: NodeDefinition<ImputationConfig> = {
  type: 'simple_imputer',
  label: 'Imputation',
  category: 'Preprocessing',
  description: 'Fill missing values in selected columns.',
  icon: PaintBucket,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Filled Data', type: 'dataset' }],
  settings: ImputationSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0,
    message: config.columns.length === 0 ? 'Select at least one column' : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
    strategy: 'mean',
  }),
};
