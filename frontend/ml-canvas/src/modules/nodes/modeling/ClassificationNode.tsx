import { Tags } from 'lucide-react';
import { TrainingSettings, TrainingConfig } from './TrainingSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';
import { StepType } from '../../../core/constants/stepTypes';

/**
 * Task-scoped Classification node (Phase 3 Part B — plan §0.6). Reuses the
 * generic `TrainingNode`'s `TrainingSettings` component verbatim (same
 * `run_mode`/CV/hyperparameter/search-space machinery), just parameterized
 * with `task="classification"` so the model dropdown only shows registry
 * items tagged `classification` instead of every non-clustering model.
 */
export const ClassificationNode = createModelingNode<TrainingConfig>({
  type: StepType.CLASSIFICATION,
  label: 'Classification',
  description: 'Train a classifier to predict a categorical target — fixed parameters or automatic tuning.',
  icon: Tags,
  settings: (props) => <TrainingSettings {...props} task="classification" />,
  defaultConfig: {
    run_mode: 'basic',
    model_type: 'random_forest_classifier',
    // Shared
    hyperparameters: {},
    cv_enabled: true,
    cv_folds: 5,
    cv_type: 'k_fold',
    cv_shuffle: true,
    cv_random_state: 42,
    cv_time_column: '',
    // Advanced-mode only
    n_trials: 10,
    metric: 'accuracy',
    search_strategy: 'random',
    random_state: 42,
    search_space: {},
  },
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }]
});
