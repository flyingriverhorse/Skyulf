import type { RegistryItem } from '../../../../core/api/registry';
import type { TrainingTask } from '../TrainingSettings';
import type { HyperparameterDef } from '../components/types';

export const TASK_TAG: Record<TrainingTask, string> = {
  classification: 'classification',
  regression: 'regression',
  text_classification: 'text',
};

/** Basic mode: a conditional param (e.g. `l1_ratio`) shows only when its fixed dependency value matches. */
export const isBasicParamVisible = (def: HyperparameterDef, hyperparameters: Record<string, unknown>): boolean => {
  if (!def.depends_on) return true;
  const { param, value } = def.depends_on;
  return (hyperparameters[param] ?? def.default) === value;
};

/** Advanced mode: a conditional param shows only when its dependency's search space includes the required value. */
export const isSearchSpaceParamVisible = (def: HyperparameterDef, searchSpace: Record<string, unknown>): boolean => {
  if (!def.depends_on) return true;
  const { param, value } = def.depends_on;
  const depValues = searchSpace[param];
  return Array.isArray(depValues) && depValues.includes(value);
};

export function isClassificationModel(task: TrainingTask | undefined, model: RegistryItem | undefined) {
  return task !== undefined ? task !== 'regression' : model?.tags?.includes('classification') ?? false;
}

export function isTrainingModel(node: RegistryItem, taskTag: string | undefined) {
  const isModeling = node.category === 'Model' || node.category === 'Modeling';
  const isClustering = node.tags?.includes('clustering') ?? false;
  if (!isModeling || isClustering) return false;
  if (taskTag) return node.tags?.includes(taskTag) ?? false;
  return true;
}

export function isGridStrategy(strategy: string | null) {
  return strategy === 'grid' || strategy === 'halving_grid';
}

export function hasLoadedSearchSpace(isNewModel: boolean, isStrategyClassChange: boolean, definitions: HyperparameterDef[]) {
  return !isNewModel && !isStrategyClassChange && definitions.length > 0;
}
