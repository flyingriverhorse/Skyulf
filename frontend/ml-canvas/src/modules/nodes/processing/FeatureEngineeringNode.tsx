import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Wand2 } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface FeatureEngineeringConfig {
  scaling: {
    method: 'standard' | 'minmax' | 'none';
    columns: string[];
  };
  encoding: {
    method: 'onehot' | 'label' | 'none';
    columns: string[];
  };
}

const FeatureEngineeringSettings: React.FC<{ config: FeatureEngineeringConfig; onChange: (c: FeatureEngineeringConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const columns = schema?.columns.map(c => c.name) || [];

  return (
    <div className="p-4 space-y-6">
      {/* Scaling Section */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold border-b pb-1">Scaling</h3>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.scaling.method}
          onChange={(e) => onChange({ ...config, scaling: { ...config.scaling, method: e.target.value as any } })}
        >
          <option value="none">None</option>
          <option value="standard">Standard Scaler</option>
          <option value="minmax">MinMax Scaler</option>
        </select>
        
        {config.scaling.method !== 'none' && (
          <div className="space-y-1 max-h-32 overflow-y-auto border rounded p-2">
            {columns.map(col => (
              <label key={`scale-${col}`} className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={config.scaling.columns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.scaling.columns, col]
                      : config.scaling.columns.filter(c => c !== col);
                    onChange({ ...config, scaling: { ...config.scaling, columns: newCols } });
                  }}
                />
                {col}
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Encoding Section */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold border-b pb-1">Encoding</h3>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.encoding.method}
          onChange={(e) => onChange({ ...config, encoding: { ...config.encoding, method: e.target.value as any } })}
        >
          <option value="none">None</option>
          <option value="onehot">One-Hot Encoding</option>
          <option value="label">Label Encoding</option>
        </select>
        
        {config.encoding.method !== 'none' && (
          <div className="space-y-1 max-h-32 overflow-y-auto border rounded p-2">
            {columns.map(col => (
              <label key={`enc-${col}`} className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={config.encoding.columns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.encoding.columns, col]
                      : config.encoding.columns.filter(c => c !== col);
                    onChange({ ...config, encoding: { ...config.encoding, columns: newCols } });
                  }}
                />
                {col}
              </label>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export const FeatureEngineeringNode: NodeDefinition<FeatureEngineeringConfig> = {
  type: 'feature_engineering',
  label: 'Feature Engineering',
  category: 'Preprocessing',
  description: 'Scale and encode features.',
  icon: Wand2,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Features', type: 'dataset' }],
  settings: FeatureEngineeringSettings,
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    scaling: { method: 'none', columns: [] },
    encoding: { method: 'none', columns: [] },
  }),
};
