import type { Node, Edge } from '@xyflow/react';
import { StepType as BackendStepType } from '../../constants/stepTypes';
import { buildBaseTuningConfig, buildFixedTrainingParams } from './training';
import type { ConvertedNode } from './types';

// Canvas node `definitionType`s that represent a trained-model spec. When one of
// these feeds an Ensemble node it acts as a *base-learner spec provider* (Phase 2):
// the ensemble reads its `model_type` + `hyperparameters` and re-fits it itself —
// sklearn Voting/Stacking always refit their base estimators, so only the recipe
// (not the fitted weights) is reused.
const MODEL_SOURCE_TYPES = new Set([
  'training',
  'classification',
  'regression',
  'text_classification',
]);

// Maps a full training-node `model_type` back to the short ensemble base-learner
// key the core resolver understands. Mirrors `_BASE_KEY_TO_REGISTRY_*` in
// `skyulf.modeling.hyperparameters._registry` (inverted). Unsupported model types
// (xgboost, lightgbm, extra_trees, …) are intentionally absent — the ensemble core
// only supports these base learners, so anything else is skipped.
const ENSEMBLE_BASE_KEY_BY_MODEL_TYPE: Record<'classification' | 'regression', Record<string, string>> = {
  classification: {
    logistic_regression: 'logistic_regression',
    random_forest_classifier: 'random_forest',
    extra_trees_classifier: 'extra_trees',
    gradient_boosting_classifier: 'gradient_boosting',
    hist_gradient_boosting_classifier: 'hist_gradient_boosting',
    adaboost_classifier: 'adaboost',
    decision_tree_classifier: 'decision_tree',
    gaussian_nb: 'gaussian_nb',
    sgd_classifier: 'sgd_classifier',
    svc: 'svc',
    k_neighbors_classifier: 'knn',
    xgboost_classifier: 'xgboost',
    lgbm_classifier: 'lightgbm',
  },
  regression: {
    linear_regression: 'linear_regression',
    ridge_regression: 'ridge',
    lasso_regression: 'lasso',
    elasticnet_regression: 'elasticnet',
    random_forest_regressor: 'random_forest',
    extra_trees_regressor: 'extra_trees',
    gradient_boosting_regressor: 'gradient_boosting',
    hist_gradient_boosting_regressor: 'hist_gradient_boosting',
    adaboost_regressor: 'adaboost',
    decision_tree_regressor: 'decision_tree',
    svr: 'svr',
    k_neighbors_regressor: 'knn',
    xgboost_regressor: 'xgboost',
    lgbm_regressor: 'lightgbm',
  },
};

export const isModelSourceType = (defType: unknown): boolean =>
  typeof defType === 'string' && MODEL_SOURCE_TYPES.has(defType);

/** Resolve a connected model node's `model_type` to an ensemble base key, or null. */
const resolveEnsembleBaseKey = (modelType: unknown, task: unknown): string | null => {
  if (typeof modelType !== 'string') return null;
  const t = task === 'regression' ? 'regression' : 'classification';
  return ENSEMBLE_BASE_KEY_BY_MODEL_TYPE[t][modelType] ?? null;
};

interface WiredBaseSpec {
  baseEstimators: string[];
  baseParams: Record<string, Record<string, unknown>>;
  modelSourceIds: Set<string>;
}

/**
 * Collect base-learner specs from the model nodes wired into an ensemble's input.
 * Returns the resolved base keys, their per-model hyperparameters, and the set of
 * source node ids (so they can be excluded from the ensemble's data `inputs`).
 */
const collectWiredBaseSpecs = (
  nodes: Node[],
  incomingEdges: Edge[],
  task: unknown,
): WiredBaseSpec => {
  const baseEstimators: string[] = [];
  const baseParams: Record<string, Record<string, unknown>> = {};
  const modelSourceIds = new Set<string>();

  for (const edge of incomingEdges) {
    const src = nodes.find((n) => n.id === edge.source);
    if (!src || !isModelSourceType(src.data.definitionType)) continue;
    modelSourceIds.add(src.id);
    const key = resolveEnsembleBaseKey(src.data.model_type, task);
    if (!key) continue;
    if (!baseEstimators.includes(key)) baseEstimators.push(key);
    const hp = src.data.hyperparameters;
    if (hasHyperparameters(hp)) {
      baseParams[key] = hp as Record<string, unknown>;
    }
  }

  return { baseEstimators, baseParams, modelSourceIds };
};

/** Only explicitly configured model parameters override the ensemble selection. */
function hasHyperparameters(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && Object.keys(value).length > 0);
}

/** Resolve model recipes, preserving the wired-model override and parameter precedence. */
function buildEnsembleStructure(data: Record<string, unknown>, wired: WiredBaseSpec): Record<string, unknown> {
  const hasWired = wired.baseEstimators.length > 0;
  const baseEstimators = hasWired ? wired.baseEstimators : data.base_estimators;
  const baseParams = hasWired
    ? { ...(data.base_estimator_params as Record<string, unknown> | undefined ?? {}), ...wired.baseParams }
    : data.base_estimator_params;
  return {
    base_estimators: baseEstimators,
    voting: data.voting,
    final_estimator: data.final_estimator,
    cv: data.cv,
    passthrough: data.passthrough,
    weights: votingWeights(data, baseEstimators),
    n_jobs: data.n_jobs,
    ...calibrationParams(data),
    base_estimator_params: baseParams,
    final_estimator_params: data.final_estimator_params,
  };
}

/** Emit voting weights only for an explicitly non-default numeric entry. */
function votingWeights(data: Record<string, unknown>, estimators: unknown): number[] | undefined {
  if (data.strategy !== 'voting') return undefined;
  const weights = data.weights as Record<string, number> | undefined;
  const bases = (estimators as string[] | undefined) ?? [];
  if (!weights || bases.length === 0) return undefined;
  const hasCustomWeight = bases.some(key => typeof weights[key] === 'number' && weights[key] !== 1);
  if (!hasCustomWeight) return undefined;
  return bases.map(key => typeof weights[key] === 'number' ? weights[key] : 1);
}

/** Regression and uncalibrated jobs omit classifier-only calibration settings. */
function calibrationParams(data: Record<string, unknown>): Record<string, unknown> {
  const enabled = data.task === 'classification' && data.calibrate_base_models === true;
  return {
    calibrate_base_models: enabled || undefined,
    calibration_method: enabled ? data.calibration_method : undefined,
    calibration_cv: enabled ? data.calibration_cv : undefined,
  };
}

/** Prefer direct data edges; otherwise inherit wired models' upstream data in order. */
function ensembleDataInputs(inputs: string[], modelIds: Set<string>, edges: Edge[]): string[] {
  if (modelIds.size === 0) return inputs;
  const directData = inputs.filter(id => !modelIds.has(id));
  if (directData.length > 0) return directData;
  const inherited = new Set<string>();
  for (const modelId of modelIds) {
    for (const edge of edges.filter(candidate => candidate.target === modelId)) {
      if (!modelIds.has(edge.source)) inherited.add(edge.source);
    }
  }
  return Array.from(inherited);
}

/** Keep fixed and tuned ensemble payloads on the shared training contract. */
function ensembleTrainingParams(data: Record<string, unknown>, structure: Record<string, unknown>): Record<string, unknown> {
  if (data.run_mode === 'advanced') {
    return {
      run_mode: 'tuned',
      target_column: data.target_column,
      algorithm: data.model_type,
      execution_mode: data.execution_mode,
      tuning_config: {
        ...buildBaseTuningConfig(data),
        ...structure,
        tune_base_models: data.tune_base_models,
      },
    };
  }
  return {
    run_mode: 'fixed',
    ...buildFixedTrainingParams(data),
    hyperparameters: structure,
  };
}

export function convertEnsembleNode(node: Node, nodes: Node[], edges: Edge[], incomingEdges: Edge[], inputs: string[]): ConvertedNode {
  const wired = collectWiredBaseSpecs(nodes, incomingEdges, node.data.task);
  const structure = buildEnsembleStructure(node.data, wired);
  return {
    stepType: BackendStepType.TRAINING,
    params: ensembleTrainingParams(node.data, structure),
    inputs: ensembleDataInputs(inputs, wired.modelSourceIds, edges),
  };
}
