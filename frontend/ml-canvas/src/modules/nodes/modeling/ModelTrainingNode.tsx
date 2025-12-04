import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { BrainCircuit } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface ModelTrainingConfig {
  targetColumn: string;
  modelType: 'random_forest' | 'logistic_regression' | 'xgboost';
  hyperparameters: {
    n_estimators?: number;
    max_depth?: number;
    learning_rate?: number;
  };
}

const ModelTrainingSettings: React.FC<{ config: ModelTrainingConfig; onChange: (c: ModelTrainingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
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
        <label className="block text-sm font-medium mb-1">Target Column</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.targetColumn}
          onChange={(e) => onChange({ ...config, targetColumn: e.target.value })}
        >
          <option value="">-- Select Target --</option>
          {columns.map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Model Type</label>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.modelType}
          onChange={(e) => onChange({ ...config, modelType: e.target.value as any })}
        >
          <option value="random_forest">Random Forest</option>
          <option value="logistic_regression">Logistic Regression</option>
          <option value="xgboost">XGBoost</option>
        </select>
      </div>

      {/* Hyperparameters based on model type */}
      {config.modelType === 'random_forest' && (
        <div className="space-y-2 border-t pt-2">
          <h4 className="text-xs font-semibold text-muted-foreground">Hyperparameters</h4>
          <div>
            <label className="block text-xs mb-1">N Estimators</label>
            <input
              type="number"
              className="w-full p-1 border rounded text-sm"
              value={config.hyperparameters.n_estimators || 100}
              onChange={(e) => onChange({ 
                ...config, 
                hyperparameters: { ...config.hyperparameters, n_estimators: parseInt(e.target.value) } 
              })}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export const ModelTrainingNode: NodeDefinition<ModelTrainingConfig> = {
  type: 'train_model_draft',
  label: 'Model Training',
  category: 'Modeling',
  description: 'Train a machine learning model.',
  icon: BrainCircuit,
  inputs: [{ id: 'train', label: 'Training Data', type: 'dataset' }],
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }],
  settings: ModelTrainingSettings,
  validate: (config) => ({ 
    isValid: !!config.targetColumn,
    message: !config.targetColumn ? 'Target column is required' : undefined
  }),
  getDefaultConfig: () => ({
    targetColumn: '',
    modelType: 'random_forest',
    hyperparameters: { n_estimators: 100 },
  }),
};
