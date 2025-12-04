import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { BarChart3 } from 'lucide-react';

interface EvaluationConfig {
  splits: string[];
  metrics: string[];
}

const EvaluationSettings: React.FC<{ config: EvaluationConfig; onChange: (c: EvaluationConfig) => void }> = ({
  config,
  onChange,
}) => {
  const availableSplits = ['train', 'validation', 'test'];
  const availableMetrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'rmse', 'r2'];

  const toggleSplit = (split: string) => {
    const newSplits = config.splits.includes(split)
      ? config.splits.filter(s => s !== split)
      : [...config.splits, split];
    onChange({ ...config, splits: newSplits });
  };

  const toggleMetric = (metric: string) => {
    const newMetrics = config.metrics.includes(metric)
      ? config.metrics.filter(m => m !== metric)
      : [...config.metrics, metric];
    onChange({ ...config, metrics: newMetrics });
  };

  return (
    <div className="p-4 space-y-6">
      <div>
        <label className="block text-sm font-medium mb-2">Evaluation Splits</label>
        <div className="space-y-2">
          {availableSplits.map(split => (
            <label key={split} className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={config.splits.includes(split)}
                onChange={() => toggleSplit(split)}
                className="rounded border-gray-300 text-primary focus:ring-primary"
              />
              <span className="capitalize">{split} Set</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Metrics to Track</label>
        <div className="grid grid-cols-2 gap-2">
          {availableMetrics.map(metric => (
            <label key={metric} className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={config.metrics.includes(metric)}
                onChange={() => toggleMetric(metric)}
                className="rounded border-gray-300 text-primary focus:ring-primary"
              />
              <span className="uppercase">{metric.replace('_', ' ')}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
};

export const EvaluationNode: NodeDefinition<EvaluationConfig> = {
  type: 'model_evaluation',
  label: 'Model Evaluation',
  category: 'Evaluation',
  description: 'Evaluate model performance on specific splits.',
  icon: BarChart3,
  inputs: [{ id: 'model', label: 'Trained Model', type: 'model' }],
  outputs: [{ id: 'report', label: 'Evaluation Report', type: 'report' }],
  settings: EvaluationSettings,
  validate: (config) => ({ 
    isValid: config.splits.length > 0,
    message: config.splits.length === 0 ? 'Select at least one split to evaluate' : undefined
  }),
  getDefaultConfig: () => ({
    splits: ['validation'],
    metrics: ['accuracy', 'f1'],
  }),
};
