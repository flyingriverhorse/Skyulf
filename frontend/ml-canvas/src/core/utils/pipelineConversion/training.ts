import { StepType as BackendStepType } from '../../constants/stepTypes';
import type { NodeConverter } from './types';

/**
 * Fixed-mode training params shared by the unified `TrainingNode` and the
 * task-scoped Classification/Regression/Text Classification nodes when
 * their `run_mode` is `'basic'`.
 */
export const buildFixedTrainingParams = (data: Record<string, unknown>): Record<string, unknown> => ({
  target_column: data.target_column,
  model_type: data.model_type,
  hyperparameters: data.hyperparameters,
  cv_enabled: data.cv_enabled,
  cv_folds: data.cv_folds,
  cv_type: data.cv_type,
  cv_shuffle: data.cv_shuffle,
  cv_random_state: data.cv_random_state,
  cv_time_column: data.cv_time_column,
  execution_mode: data.execution_mode,
});

/**
 * The tuning-config fields common to every tuning-engine consumer (the plain
 * `tuning` node, the unified `TrainingNode`
 * in advanced mode, and `EnsembleNode`'s advanced mode). Callers add their
 * own structural fields on top (`search_space` for plain training nodes,
 * `base_estimators`/`final_estimator`/etc. for the ensemble).
 */
export const buildBaseTuningConfig = (data: Record<string, unknown>): Record<string, unknown> => ({
  strategy: data.search_strategy,
  strategy_params: data.strategy_params || {},
  metric: data.metric,
  n_trials: data.n_trials,
  cv_enabled: data.cv_enabled,
  cv_folds: data.cv_folds,
  cv_type: data.cv_type,
  cv_shuffle: data.cv_shuffle,
  cv_random_state: data.cv_random_state,
  cv_time_column: data.cv_time_column,
  random_state: data.random_state,
  tune_threshold: data.tune_threshold ?? false,
});

export const convertTrainingNode: NodeConverter = (node) => {
  const stepType = BackendStepType.TRAINING;
  let params: Record<string, unknown> = {};
  // The generic TrainingNode and the 3 task-scoped Classification/
  // Regression/Text Classification nodes all dispatch through the
  // same fixed/tuned param-building helpers and emit the canonical
  // `training` step_type.
  const isAdvanced = node.data.run_mode === 'advanced';
  if (isAdvanced) {
    params = {
      run_mode: 'tuned',
      target_column: node.data.target_column,
      algorithm: node.data.model_type,
      execution_mode: node.data.execution_mode,
      tuning_config: {
        ...buildBaseTuningConfig(node.data),
        search_space: node.data.search_space
      }
    };
  } else {
    params = {
      run_mode: 'fixed',
      ...buildFixedTrainingParams(node.data)
    };
  }
  return { stepType, params };
};

export const convertSegmentationNode: NodeConverter = (node) => {
  const stepType = BackendStepType.TRAINING;
  const params = {
    run_mode: 'fixed',
    // No target_column — clustering is unsupervised. The backend
    // treats an empty string as the "no target" sentinel.
    target_column: '',
    model_type: node.data.model_type,
    hyperparameters: node.data.hyperparameters,
    cv_enabled: false,
    execution_mode: node.data.execution_mode,
    // Optional column (e.g. species name) excluded from training
    // but kept for post-hoc cluster interpretation — see
    // `reference_crosstab` in the evaluation report.
    reference_column: node.data.reference_column || undefined,
  };
  return { stepType, params };
};
