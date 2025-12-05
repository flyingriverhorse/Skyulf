import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Filter } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface FeatureSelectionConfig {
  method: 'variance_threshold' | 'select_k_best' | 'select_percentile' | 'select_from_model' | 'rfe';
  threshold?: number;
  k?: number;
  percentile?: number;
  estimator?: 'RandomForest' | 'LogisticRegression' | 'LinearRegression';
  target_column?: string;
}

const FeatureSelectionSettings: React.FC<{ config: FeatureSelectionConfig; onChange: (c: FeatureSelectionConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium">Selection Method</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.method}
          onChange={(e) => onChange({ ...config, method: e.target.value as any })}
        >
          <option value="variance_threshold">Variance Threshold</option>
          <option value="select_k_best">Select K Best</option>
          <option value="select_percentile">Select Percentile</option>
          <option value="select_from_model">Select From Model</option>
          <option value="rfe">Recursive Feature Elimination (RFE)</option>
        </select>
      </div>

      {config.method === 'variance_threshold' && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Threshold</label>
          <input
            type="number"
            step="0.01"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.threshold ?? 0}
            onChange={(e) => onChange({ ...config, threshold: parseFloat(e.target.value) })}
          />
          <p className="text-xs text-muted-foreground">Features with variance lower than this will be removed.</p>
        </div>
      )}

      {config.method === 'select_k_best' && (
        <div className="space-y-2">
          <label className="text-sm font-medium">K (Number of Features)</label>
          <input
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.k ?? 10}
            onChange={(e) => onChange({ ...config, k: parseInt(e.target.value) })}
          />
        </div>
      )}

      {config.method === 'select_percentile' && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Percentile</label>
          <input
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.percentile ?? 10}
            onChange={(e) => onChange({ ...config, percentile: parseInt(e.target.value) })}
          />
        </div>
      )}

      {(config.method === 'select_from_model' || config.method === 'rfe') && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Estimator</label>
          <select
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.estimator ?? 'RandomForest'}
            onChange={(e) => onChange({ ...config, estimator: e.target.value as any })}
          >
            <option value="RandomForest">Random Forest</option>
            <option value="LogisticRegression">Logistic Regression</option>
            <option value="LinearRegression">Linear Regression</option>
          </select>
        </div>
      )}

      {config.method !== 'variance_threshold' && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Target Column</label>
          <select
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.target_column ?? ''}
            onChange={(e) => onChange({ ...config, target_column: e.target.value })}
          >
            <option value="">Select Target...</option>
            {columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <p className="text-xs text-muted-foreground">Required for supervised selection methods.</p>
        </div>
      )}
    </div>
  );
};

export const FeatureSelectionNode: NodeDefinition<FeatureSelectionConfig> = {
  type: 'feature_selection',
  label: 'Feature Selection',
  category: 'Preprocessing',
  description: 'Select the most important features.',
  icon: Filter,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Selected', type: 'dataset' }],
  settings: FeatureSelectionSettings,
  validate: (config) => {
    if (config.method !== 'variance_threshold' && !config.target_column) {
      return { isValid: false, message: 'Target column is required for this method.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'select_k_best',
    k: 10,
  }),
};
