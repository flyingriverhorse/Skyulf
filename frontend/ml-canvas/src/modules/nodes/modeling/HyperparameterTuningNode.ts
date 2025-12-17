import { NodeDefinition } from '../../../core/types/nodes';
import { Sliders } from 'lucide-react';
import { TuningSettings } from './HyperparameterTuningSettings';

export const HyperparameterTuningNode: NodeDefinition = {
  type: 'hyperparameter_tuning',
  label: 'Train Model and Optimize',
  category: 'Modeling',
  description: 'Automatically optimize model performance.',
  icon: Sliders,
  inputs: [{ id: 'in', label: 'Training Data', type: 'dataset' }],
  outputs: [{ id: 'model', label: 'Best Model', type: 'model' }],
  settings: TuningSettings,
  validate: (config) => {
    if (!config.target_column) return { isValid: false, message: 'Target column is required.' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    target_column: '',
    model_type: 'random_forest_classifier',
    n_trials: 10,
    metric: 'accuracy',
    search_strategy: 'random',
    cv_enabled: true,
    cv_folds: 5,
    cv_type: 'k_fold',
    cv_shuffle: true,
    cv_random_state: 42,
    random_state: 42,
    search_space: {}
  })
};
