import { FileText } from 'lucide-react';
import { TrainingSettings, TrainingConfig } from './TrainingSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';
import { StepType } from '../../../core/constants/stepTypes';

/**
 * Task-scoped Text Classification node (Phase 3 Part B — plan §0.6). Reuses
 * the generic `TrainingNode`'s `TrainingSettings` component verbatim, just
 * parameterized with `task="text_classification"` so the model dropdown only
 * shows registry items tagged `text` (Naive Bayes / Logistic Regression /
 * SGD — models suited to vectorized text features), instead of every
 * non-clustering model. Expects a vectorizer node (TF-IDF/Count/Hashing)
 * wired upstream, same as the existing text-classification canvas template.
 */
export const TextClassificationNode = createModelingNode<TrainingConfig>({
  type: StepType.TEXT_CLASSIFICATION,
  label: 'Text Classification',
  description: 'Train a classifier on vectorized text features — fixed parameters or automatic tuning.',
  icon: FileText,
  settings: (props) => <TrainingSettings {...props} task="text_classification" />,
  defaultConfig: {
    run_mode: 'basic',
    model_type: 'multinomial_nb',
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
