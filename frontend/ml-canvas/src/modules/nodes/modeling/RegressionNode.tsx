import { TrendingUp } from 'lucide-react';
import { TrainingSettings, TrainingConfig } from './TrainingSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';
import { StepType } from '../../../core/constants/stepTypes';

/**
 * Task-scoped Regression node (Phase 3 Part B — plan §0.6). Reuses the
 * generic `TrainingNode`'s `TrainingSettings` component verbatim, just
 * parameterized with `task="regression"` so the model dropdown only shows
 * registry items tagged `regression` instead of every non-clustering model.
 */
export const RegressionNode = createModelingNode<TrainingConfig>({
  type: StepType.REGRESSION,
  label: 'Regression',
  description: 'Train a regressor to predict a numeric target — fixed parameters or automatic tuning.',
  icon: TrendingUp,
  settings: (props) => <TrainingSettings {...props} task="regression" />,
  defaultConfig: {
    run_mode: 'basic',
    model_type: 'random_forest_regressor',
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
    metric: 'r2',
    search_strategy: 'random',
    random_state: 42,
    search_space: {},
  },
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }]
});
