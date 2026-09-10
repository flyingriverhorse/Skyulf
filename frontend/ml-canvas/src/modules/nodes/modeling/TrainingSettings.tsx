import React from 'react';
import type { ExecutionMode } from '../../../core/types/executionMode';
import { useTrainingSettings } from './trainingSettings/useTrainingSettings';
import { ModelConfigSection } from './trainingSettings/ModelConfigSection';
import { HyperparametersSection, SearchSpaceSection } from './trainingSettings/ParameterSections';
import { TrainingDialogs } from './trainingSettings/TrainingDialogs';
import { TrainingFooter } from './trainingSettings/TrainingFooter';
import { TrainingInformation } from './trainingSettings/TrainingInformation';
import { TrainingLayout } from './trainingSettings/TrainingLayout';

export type TrainingRunMode = 'basic' | 'advanced';

export interface TrainingConfig {
  run_mode: TrainingRunMode;
  target_column: string;
  model_type: string;
  // Basic-mode fixed hyperparameters.
  hyperparameters: Record<string, unknown>;
  // Advanced-mode tuning fields.
  search_space: Record<string, unknown>;
  n_trials: number;
  metric: string;
  search_strategy: string;
  strategy_params?: Record<string, unknown>;
  random_state: number;
  // F-13: tune the decision threshold on the validation split after tuning
  // (binary classification only; off by default).
  tune_threshold?: boolean;
  // Shared CV section.
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
  cv_time_column?: string;
  execution_mode?: ExecutionMode;
}

/**
 * Task types the 4 dedicated task-scoped nodes (Phase 3 Part B, plan §0.6)
 * can filter the model dropdown by. `undefined` (the generic `TrainingNode`)
 * keeps the original behavior: every non-clustering model. Maps to registry
 * tags added on the backend: `classification`/`regression` tag names match
 * directly; `text_classification` maps to the `text` tag (models suited for
 * vectorized text features, e.g. Naive Bayes / Logistic Regression / SGD).
 */
export type TrainingTask = 'classification' | 'regression' | 'text_classification';

export const TrainingSettings: React.FC<{
  config: TrainingConfig;
  onChange: (c: TrainingConfig) => void;
  nodeId?: string;
  /** Restricts the model dropdown to models tagged for this task. Omit for
   * the generic `TrainingNode`, which shows every non-clustering model. */
  task?: TrainingTask;
}> = ({
  config,
  onChange,
  nodeId,
  task,
}) => {
  const state = useTrainingSettings(config, onChange, nodeId, task);
  const secondaryPanel = state.isAdvanced
    ? <SearchSpaceSection {...state} />
    : <HyperparametersSection {...state} />;

  return (
    <div className="flex flex-col h-full" ref={state.containerRef}>
      <TrainingInformation {...state} />
      <TrainingDialogs {...state} />
      <TrainingLayout {...state} modelPanel={<ModelConfigSection {...state} />} secondaryPanel={secondaryPanel} />
      <TrainingFooter {...state} />
    </div>
  );
};
