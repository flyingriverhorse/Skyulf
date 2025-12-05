import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Hash } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface LabelEncodingConfig {
  columns: string[];
}

const LabelEncodingSettings: React.FC<{ config: LabelEncodingConfig; onChange: (c: LabelEncodingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  
  const categoricalColumns = schema 
    ? Object.values(schema.columns)
        .filter(c => ['object', 'string', 'category', 'bool'].some(t => c.dtype?.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];

  return (
    <div className="p-4 space-y-4">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
          Connect a dataset node to see available columns.
        </div>
      )}
      
      <div>
        <label className="block text-sm font-medium mb-2">Categorical Columns</label>
        {categoricalColumns.length > 0 ? (
          <div className="space-y-1 max-h-60 overflow-y-auto border rounded p-2 bg-background">
            {categoricalColumns.map(col => (
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
            {isLoading ? 'Loading schema...' : 'No categorical columns found'}
          </div>
        )}
        <p className="text-[10px] text-muted-foreground mt-2">
          Converts categories to integer codes (0, 1, 2...). Best for ordinal data.
        </p>
      </div>
    </div>
  );
};

export const LabelEncodingNode: NodeDefinition<LabelEncodingConfig> = {
  type: 'label_encoding',
  label: 'Label Encoding',
  category: 'Preprocessing',
  description: 'Convert categorical variables into integer codes.',
  icon: Hash,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Encoded Data', type: 'dataset' }],
  settings: LabelEncodingSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0,
    message: config.columns.length === 0 ? 'Select at least one column' : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
  }),
};
