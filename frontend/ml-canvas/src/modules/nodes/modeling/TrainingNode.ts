import { BrainCircuit } from 'lucide-react';
import { TrainingSettings, TrainingConfig } from './TrainingSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';
import { StepType } from '../../../core/constants/stepTypes';

/**
 * Unified training node (Phase 3 of the Basic Training / Advanced Tuning
 * merge — see `temp/processing/basic_training_advanced_tuning_unification_plan.md`).
 * Combines the former `BasicTrainingNode` and `AdvancedTuningNode` into a
 * single canvas node with a `run_mode: 'basic' | 'advanced'` toggle, mirroring
 * `EnsembleNode`'s existing pattern. `BasicTrainingNode`/`AdvancedTuningNode`
 * remain registered separately (unchanged) purely so previously saved
 * canvases that reference those exact `type` strings keep loading.
 */
export const TrainingNode = createModelingNode<TrainingConfig>({
  type: StepType.TRAINING,
  label: 'Training',
  description: 'Train a model with fixed parameters, or automatically tune it.',
  icon: BrainCircuit,
  settings: TrainingSettings,
  defaultConfig: {
    run_mode: 'basic',
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
