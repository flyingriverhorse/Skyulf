import { Boxes } from 'lucide-react';
import { EnsembleSettings, EnsembleConfig } from './EnsembleSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';

/**
 * Dedicated ensemble model node. Combines several base learners via Voting or
 * Stacking (classification or regression). Reuses the shared training plumbing
 * (`useTrainingNodeContext`) and runs either through the standard
 * `basic_training` path or, in advanced mode, the `advanced_tuning` engine —
 * which can also tune the base models' hyperparameters via nested search keys.
 */
export const EnsembleNode = createModelingNode({
  type: 'EnsembleNode',
  label: 'Ensemble',
  description: 'Combine multiple models via Voting or Stacking.',
  icon: Boxes,
  settings: EnsembleSettings,
  defaultConfig: {
    task: 'classification',
    strategy: 'voting',
    model_type: 'voting_classifier',
    base_estimators: ['random_forest', 'logistic_regression', 'gradient_boosting'],
    voting: 'soft',
    final_estimator: 'logistic_regression',
    cv: 5,
    passthrough: false,
    n_jobs: 1,
    calibrate_base_models: false,
    calibration_method: 'sigmoid',
    calibration_cv: 3,
    cv_enabled: true,
    cv_folds: 5,
    cv_type: 'k_fold',
    cv_shuffle: true,
    cv_random_state: 42,
    cv_time_column: '',
    base_estimator_params: {},
    final_estimator_params: {},
    run_mode: 'basic',
    search_strategy: 'random',
    n_trials: 20,
    metric: 'accuracy',
    tune_base_models: true,
    random_state: 42,
  },
  bodyPreview: (config: EnsembleConfig) => {
    const n = config.base_estimators?.length ?? 0;
    const strategy = config.strategy ?? 'voting';
    if (!n) return strategy;
    return `${strategy} \u00b7 ${n} ${n === 1 ? 'model' : 'models'}`;
  },
  validate: (config: EnsembleConfig) => {
    if (!config.target_column) return { isValid: false, message: 'Target column is required.' };
    if ((config.base_estimators?.length ?? 0) < 2)
      return { isValid: false, message: 'Select at least two base models.' };
    return { isValid: true };
  },
  outputs: [{ id: 'model', label: 'Ensemble Model', type: 'model' }],
});
