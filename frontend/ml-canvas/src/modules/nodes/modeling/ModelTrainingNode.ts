import { NodeDefinition } from '../../../core/types/nodes';
import { BrainCircuit } from 'lucide-react';
import { ModelTrainingSettings } from './ModelTrainingSettings';

export const ModelTrainingNode: NodeDefinition = {
  type: 'model_training',
  label: 'Standard Training',
  category: 'Modeling',
  description: 'Train a model with fixed or default parameters.',
  icon: BrainCircuit,
  inputs: [{ id: 'in', label: 'Training Data', type: 'dataset' }],
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }],
  settings: ModelTrainingSettings,
  validate: (config) => {
    if (!config.target_column) return { isValid: false, message: 'Target column is required.' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    target_column: '',
    model_type: 'random_forest_classifier',
    hyperparameters: {},
    cv_enabled: true,
    cv_folds: 5,
    cv_type: 'k_fold',
    cv_shuffle: true,
    cv_random_state: 42
  })
};
