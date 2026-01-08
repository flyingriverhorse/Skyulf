import { Sliders } from 'lucide-react';
import { AdvancedTuningSettings } from './AdvancedTuningSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';

export const AdvancedTuningNode = createModelingNode({
  type: 'advanced_tuning',
  label: 'Advanced Tuning and Training',
  description: 'Automatically optimize model performance.',
  icon: Sliders,
  settings: AdvancedTuningSettings,
  defaultConfig: {
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
  },
  outputs: [{ id: 'model', label: 'Best Model', type: 'model' }]
});
