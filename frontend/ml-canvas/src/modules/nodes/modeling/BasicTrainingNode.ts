import { BrainCircuit } from 'lucide-react';
import { BasicTrainingSettings } from './BasicTrainingSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';

export const BasicTrainingNode = createModelingNode({
  type: 'basic_training',
  label: 'Basic Training',
  description: 'Train a model with fixed or default parameters.',
  icon: BrainCircuit,
  settings: BasicTrainingSettings,
  defaultConfig: {
    hyperparameters: {},
    cv_enabled: true,
    cv_folds: 5,
    cv_type: 'k_fold',
    cv_shuffle: true,
    cv_random_state: 42
  },
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }]
});
