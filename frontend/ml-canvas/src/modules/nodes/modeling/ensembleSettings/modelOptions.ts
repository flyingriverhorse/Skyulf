import type { ColumnProfile } from '../../../../core/api/client';
import type { EnsembleConfig } from '../EnsembleSettings';

export type Task = EnsembleConfig['task'];
export type Strategy = EnsembleConfig['strategy'];
export type RunMode = EnsembleConfig['run_mode'];
export type UpdateFn = (patch: Partial<EnsembleConfig>) => void;

export type Option = { label: string; value: string };

// Base learners mirrored from skyulf.modeling.ensemble (BASE_ESTIMATORS_*).
// Optional boosters (xgboost / lightgbm) are omitted from this manual picker
// (optional wheels) but are auto-detected when wired in from a model node.
const CLF_OPTIONS: Option[] = [
  { label: 'Logistic Regression', value: 'logistic_regression' },
  { label: 'Random Forest', value: 'random_forest' },
  { label: 'Extra Trees', value: 'extra_trees' },
  { label: 'Gradient Boosting', value: 'gradient_boosting' },
  { label: 'Hist Gradient Boosting', value: 'hist_gradient_boosting' },
  { label: 'AdaBoost', value: 'adaboost' },
  { label: 'Decision Tree', value: 'decision_tree' },
  { label: 'Gaussian Naive Bayes', value: 'gaussian_nb' },
  { label: 'SGD Classifier', value: 'sgd_classifier' },
  { label: 'Support Vector Classifier', value: 'svc' },
  { label: 'K-Nearest Neighbors', value: 'knn' },
];

const REG_OPTIONS: Option[] = [
  { label: 'Linear Regression', value: 'linear_regression' },
  { label: 'Ridge', value: 'ridge' },
  { label: 'Lasso', value: 'lasso' },
  { label: 'ElasticNet', value: 'elasticnet' },
  { label: 'Random Forest', value: 'random_forest' },
  { label: 'Extra Trees', value: 'extra_trees' },
  { label: 'Gradient Boosting', value: 'gradient_boosting' },
  { label: 'Hist Gradient Boosting', value: 'hist_gradient_boosting' },
  { label: 'AdaBoost', value: 'adaboost' },
  { label: 'Decision Tree', value: 'decision_tree' },
  { label: 'Support Vector Regressor', value: 'svr' },
  { label: 'K-Nearest Neighbors', value: 'knn' },
];

export function baseOptions(task: Task, availableIds?: Set<string>, currentSelection: string[] = []): Option[] {
  const base = task === 'classification' ? [...CLF_OPTIONS] : [...REG_OPTIONS];
  const suffix = task === 'classification' ? 'classifier' : 'regressor';
  const optionalModels = [
    { label: 'XGBoost', value: 'xgboost', registryId: `xgboost_${suffix}` },
    { label: 'LightGBM', value: 'lightgbm', registryId: `lgbm_${suffix}` },
  ];
  for (const { label, value, registryId } of optionalModels) {
    if (availableIds?.has(registryId) || currentSelection.includes(value)) {
      base.push({ label, value });
    }
  }
  return base;
}

export function resolveModelId(task: Task, strategy: Strategy): string {
  return `${strategy}_${task === 'classification' ? 'classifier' : 'regressor'}`;
}

export function defaultBaseEstimators(task: Task): string[] {
  return task === 'classification'
    ? ['random_forest', 'logistic_regression', 'gradient_boosting']
    : ['random_forest', 'gradient_boosting', 'ridge'];
}

export function defaultFinalEstimator(task: Task): string {
  return task === 'classification' ? 'logistic_regression' : 'ridge';
}

export function defaultMetric(task: Task): string {
  return task === 'classification' ? 'accuracy' : 'r2';
}

/**
 * Infer the ML task from the target column's profile, mirroring the backend EDA
 * heuristic (`skyulf.profiling`): float → regression, boolean / string /
 * categorical → classification, integer → classification only when it is
 * low-cardinality (reads as discrete class labels) otherwise regression.
 * Returns ``null`` when the dtype is unknown so callers keep the current task.
 */
export function inferTaskFromColumn(col: ColumnProfile | undefined): Task | null {
  if (!col) return null;
  const dt = String(col.dtype).toLowerCase();
  if (dt.includes('bool')) return 'classification';
  if (['object', 'string', 'category', 'text'].some((type) => dt.includes(type))) {
    return 'classification';
  }
  if (['float', 'double', 'decimal'].some((type) => dt.includes(type))) {
    return 'regression';
  }
  if (dt.includes('int')) {
    return col.unique_count > 0 && col.unique_count <= 20 ? 'classification' : 'regression';
  }
  return null;
}

export function metricOptions(task: Task): Option[] {
  return task === 'classification'
    ? [
        { label: 'Accuracy', value: 'accuracy' },
        { label: 'F1', value: 'f1' },
        { label: 'Balanced Accuracy', value: 'balanced_accuracy' },
        { label: 'ROC AUC', value: 'roc_auc' },
      ]
    : [
        { label: 'R²', value: 'r2' },
        { label: 'RMSE', value: 'rmse' },
        { label: 'MAE', value: 'mae' },
      ];
}

export const SEARCH_STRATEGIES: Option[] = [
  { label: 'Random', value: 'random' },
  { label: 'Grid', value: 'grid' },
  { label: 'Optuna', value: 'optuna' },
  { label: 'Halving Random', value: 'halving_random' },
  { label: 'Halving Grid', value: 'halving_grid' },
];

const LOOKUP_BASE_KEY_BY_MODEL_TYPE: Record<'classification' | 'regression', Record<string, string>> = {
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

export function lookupTaskFromModelType(modelType: string): 'classification' | 'regression' | null {
  if (modelType in LOOKUP_BASE_KEY_BY_MODEL_TYPE.classification) return 'classification';
  if (modelType in LOOKUP_BASE_KEY_BY_MODEL_TYPE.regression) return 'regression';
  if (modelType.endsWith('_classifier') || ['logistic_regression', 'gaussian_nb', 'sgd_classifier', 'svc'].includes(modelType)) {
    return 'classification';
  }
  if (modelType.endsWith('_regressor') || modelType.includes('regression') || ['linear_regression', 'svr'].includes(modelType)) {
    return 'regression';
  }
  return null;
}

export function resolveEnsembleBaseKey(modelType: string, task: Task): string | null {
  return LOOKUP_BASE_KEY_BY_MODEL_TYPE[task][modelType] ?? null;
}

export function optionLabelMap(options: Option[]): Record<string, string> {
  return Object.fromEntries(options.map((o) => [o.value, o.label]));
}
