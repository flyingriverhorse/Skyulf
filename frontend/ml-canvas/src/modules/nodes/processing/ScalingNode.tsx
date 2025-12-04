import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Scaling } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface ScalingConfig {
  columns: string[];
  method: 'standard' | 'minmax' | 'maxabs' | 'robust';
}

const ScalingSettings: React.FC<{ config: ScalingConfig; onChange: (c: ScalingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  
  // Filter for numeric columns only, as scaling only applies to them
  const numericColumns = schema?.columns
    .filter(c => ['int', 'float', 'number'].some(t => c.dtype?.toLowerCase().includes(t)))
    .map(c => c.name) || [];

  return (
    <div className="p-4 space-y-4">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
          Connect a dataset node to see available columns.
        </div>
      )}
      
      <div>
        <label className="block text-sm font-medium mb-1">Scaling Method</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.method}
          onChange={(e) => onChange({ ...config, method: e.target.value as any })}
        >
          <option value="standard">Standard Scaler (Z-Score)</option>
          <option value="minmax">MinMax Scaler (0-1)</option>
          <option value="maxabs">MaxAbs Scaler</option>
          <option value="robust">Robust Scaler (Outlier Safe)</option>
        </select>
        <p className="text-[10px] text-muted-foreground mt-1">
          {config.method === 'standard' && 'Centers data around 0 with unit variance.'}
          {config.method === 'minmax' && 'Scales data to a fixed range [0, 1].'}
          {config.method === 'maxabs' && 'Scales data by its maximum absolute value.'}
          {config.method === 'robust' && 'Scales data using statistics that are robust to outliers.'}
        </p>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Numeric Columns</label>
        {numericColumns.length > 0 ? (
          <div className="space-y-1 max-h-60 overflow-y-auto border rounded p-2 bg-background">
            {numericColumns.map(col => (
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
            {isLoading ? 'Loading schema...' : 'No numeric columns found'}
          </div>
        )}
      </div>
    </div>
  );
};

export const ScalingNode: NodeDefinition<ScalingConfig> = {
  type: 'scale_numeric_features',
  label: 'Scaling',
  category: 'Preprocessing',
  description: 'Scale numeric features to a standard range.',
  icon: Scaling, // Note: You might need to import a real icon or use a placeholder if 'Scaling' doesn't exist in lucide-react
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Scaled Data', type: 'dataset' }],
  settings: ScalingSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0,
    message: config.columns.length === 0 ? 'Select at least one column' : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
    method: 'standard',
  }),
};
