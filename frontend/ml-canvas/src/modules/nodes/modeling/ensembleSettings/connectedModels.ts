import type { Edge, Node } from '@xyflow/react';
import type { EnsembleConfig } from '../EnsembleSettings';
import { lookupTaskFromModelType, resolveEnsembleBaseKey, resolveModelId, type Task } from './modelOptions';

type ModelData = Partial<EnsembleConfig> & {
  definitionType?: string;
  hyperparameters?: Record<string, unknown>;
  search_space?: Record<string, unknown>;
  params?: Record<string, unknown>;
};

/** Preserve incoming-edge order: it determines task, target and CV precedence. */
export function incomingModelNodes(nodeId: string | undefined, nodes: Node[], edges: Edge[]): Node<ModelData>[] {
  if (!nodeId) return [];
  const incoming: Node<ModelData>[] = [];
  for (const edge of edges.filter((edge) => edge.target === nodeId)) {
    const source = nodes.find((node) => node.id === edge.source);
    if (source && ['training', 'classification', 'regression', 'text_classification'].includes(source.data.definitionType as string)) {
      incoming.push(source as Node<ModelData>);
    }
  }
  return incoming;
}

function connectedTaskPatch(models: Node<ModelData>[], config: EnsembleConfig): Partial<EnsembleConfig> {
  if (config.task_manual) return {};
  for (const { data } of models) {
    if (!data.model_type) continue;
    const task = lookupTaskFromModelType(data.model_type);
    if (!task) continue;
    if (task === config.task) return {};
    return { task, model_type: resolveModelId(task, config.strategy) };
  }
  return {};
}

function connectedTuning(models: Node<ModelData>[], config: EnsembleConfig): Partial<EnsembleConfig> {
  const advanced = models.find((model) => model.data.run_mode === 'advanced');
  if (!advanced) return { run_mode: 'basic' };
  return {
    run_mode: 'advanced',
    search_strategy: advanced.data.search_strategy || config.search_strategy,
    n_trials: advanced.data.n_trials || config.n_trials,
    metric: advanced.data.metric || config.metric,
  };
}

function connectedCrossValidation(data: ModelData, config: EnsembleConfig): Partial<EnsembleConfig> {
  const timeColumn = data.cv_time_column || config.cv_time_column;
  return {
    cv_enabled: data.cv_enabled !== undefined ? data.cv_enabled : config.cv_enabled,
    cv_folds: data.cv_folds || config.cv_folds,
    cv_type: data.cv_type || config.cv_type,
    cv_shuffle: data.cv_shuffle !== undefined ? data.cv_shuffle : config.cv_shuffle,
    cv_random_state: data.cv_random_state ?? config.cv_random_state,
    ...(timeColumn !== undefined ? { cv_time_column: timeColumn } : {}),
  };
}

function mergeBaseParameters(data: ModelData, key: string, params: NonNullable<EnsembleConfig['base_estimator_params']>) {
  const hyperparameters = data.hyperparameters || data.search_space || data.params;
  if (hyperparameters && typeof hyperparameters === 'object' && Object.keys(hyperparameters).length > 0) {
    params[key] = { ...(params[key] || {}), ...hyperparameters };
  }
}

function connectedBases(models: Node<ModelData>[], task: Task, config: EnsembleConfig) {
  const estimators: string[] = [];
  const params = { ...(config.base_estimator_params || {}) };
  for (const { data } of models) {
    if (!data.model_type) continue;
    const key = resolveEnsembleBaseKey(data.model_type, task);
    if (!key) continue;
    if (!estimators.includes(key)) estimators.push(key);
    mergeBaseParameters(data, key, params);
  }
  return { estimators, params };
}

function baseSelectionChanged(estimators: string[], config: EnsembleConfig): boolean {
  return estimators.length > 0 &&
    (estimators.length !== config.base_estimators?.length ||
      !estimators.every((value) => config.base_estimators?.includes(value)));
}

/** Compare only derived fields; parameter-only differences historically do not trigger sync. */
function changedFields(config: EnsembleConfig, candidate: Partial<EnsembleConfig>): Partial<EnsembleConfig> {
  return Object.fromEntries(Object.entries(candidate).filter(([key, value]) => config[key as keyof EnsembleConfig] !== value));
}

/** Derive the existing connected-model update without changing effect timing or dependencies. */
export function connectedModelPatch(models: Node<ModelData>[], config: EnsembleConfig): Partial<EnsembleConfig> | null {
  const firstModel = models[0];
  if (!firstModel) return null;
  const taskPatch = connectedTaskPatch(models, config);
  const { estimators, params } = connectedBases(models, taskPatch.task ?? config.task, config);
  const candidate = {
    ...taskPatch,
    ...connectedTuning(models, config),
    ...connectedCrossValidation(firstModel.data, config),
  };
  if (baseSelectionChanged(estimators, config)) candidate.base_estimators = estimators;
  const target = firstModel.data.target_column || config.target_column;
  if (target) candidate.target_column = target;
  const patch = changedFields(config, candidate);
  if (Object.keys(patch).length === 0) return null;
  return { ...patch, base_estimator_params: params };
}
