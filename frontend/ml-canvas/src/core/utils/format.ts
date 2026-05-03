const METRIC_BASE_DESCRIPTIONS: Record<string, string> = {
  accuracy: 'Fraction of correctly classified samples. Can be misleading when classes are imbalanced.',
  f1: 'Harmonic mean of precision and recall. Balances both. Use when false positives and negatives both matter.',
  f1_weighted: 'F1 averaged across classes weighted by class support. Good choice for imbalanced datasets.',
  f1_macro: 'F1 averaged across classes with equal weight regardless of class size.',
  f1_micro: 'F1 computed globally by pooling true/false positives across all classes.',
  roc_auc: 'Area under the ROC curve. 1.0 = perfect classifier, 0.5 = random guess. Threshold-independent.',
  roc_auc_weighted: 'ROC AUC averaged across classes weighted by support.',
  roc_auc_ovr: 'ROC AUC using one-vs-rest strategy.',
  roc_auc_ovo: 'ROC AUC using one-vs-one strategy.',
  roc_auc_ovr_weighted: 'ROC AUC (one-vs-rest, weighted by class support) for multiclass. Higher = better.',
  roc_auc_ovo_weighted: 'ROC AUC (one-vs-one, weighted by class support) for multiclass. Higher = better.',
  precision: 'Of all positive predictions made, how many were actually correct.',
  precision_weighted: 'Precision averaged by class support.',
  recall: 'Of all actual positive samples, how many did the model correctly identify.',
  recall_weighted: 'Recall averaged by class support.',
  r2: 'R² score: fraction of variance explained. 1.0 = perfect, 0 = mean-only baseline, <0 = worse than predicting the mean.',
  rmse: 'Root mean squared error. Same units as the target. Penalises large errors more than MAE.',
  mae: 'Mean absolute error. Lower is better. More robust to outliers than RMSE.',
  mse: 'Mean squared error. Squares errors so large deviations are penalised heavily.',
  log_loss: 'Negative log-likelihood (log loss). Lower is better. Penalises confident wrong predictions.',
  g_score: 'Geometric mean of sensitivity and specificity. Useful for imbalanced classification.',
  balanced_accuracy: 'Accuracy averaged equally across classes. Unlike standard accuracy, not skewed by majority class size.',
  matthews_corrcoef: 'Matthews Correlation Coefficient (MCC). Single number summary of a confusion matrix. +1 = perfect, 0 = random, -1 = inverse. Best metric for imbalanced binary classification.',
  pr_auc: 'Area under Precision-Recall curve. More informative than ROC AUC when the positive class is rare.',
  pr_auc_weighted: 'PR AUC averaged across classes by support.',
  mape: 'Mean absolute percentage error. Expresses average error as a % of the true value. Avoid when targets are near zero.',
  explained_variance: 'Fraction of target variance explained by the model. Like R² but does not penalise systematic bias.',
  best_score: 'Best cross-validation score found across all tuning trials. This value drove hyperparameter selection.',
};

const METRIC_PREFIX_CONTEXT: Record<string, string> = {
  train: 'Train set —',
  test: 'Test set (held-out) —',
  val: 'Validation set —',
  cv: 'Cross-validation mean —',
};

/**
 * Returns a short human-readable description for a metric key such as
 * "test_f1_weighted", "cv_accuracy_mean", "cv_accuracy_std", "best_score".
 */
export function getMetricDescription(key: string): string | undefined {
  if (key === 'best_score') return METRIC_BASE_DESCRIPTIONS.best_score;

  // cv_*_std: special description
  if (/^cv_.+_std$/.test(key)) {
    const base = key.replace(/^cv_/, '').replace(/_std$/, '');
    const desc = METRIC_BASE_DESCRIPTIONS[base];
    return desc
      ? `Std deviation of ${base} across CV folds — lower means more consistent results. ${desc}`
      : `Std deviation of ${base} across cross-validation folds. Lower = more consistent.`;
  }

  let prefix = '';
  let stripped = key;
  if (key.startsWith('train_')) { prefix = 'train'; stripped = key.replace('train_', ''); }
  else if (key.startsWith('test_')) { prefix = 'test'; stripped = key.replace('test_', ''); }
  else if (key.startsWith('val_')) { prefix = 'val'; stripped = key.replace('val_', ''); }
  else if (key.startsWith('cv_')) { prefix = 'cv'; stripped = key.replace('cv_', '').replace(/_mean$/, ''); }

  const baseDesc = METRIC_BASE_DESCRIPTIONS[stripped];
  const ctx = METRIC_PREFIX_CONTEXT[prefix];
  if (baseDesc && ctx) return `${ctx} ${baseDesc}`;
  if (baseDesc) return baseDesc;
  if (ctx) return ctx;
  return undefined;
}

const HYPERPARAM_DESCRIPTIONS: Record<string, string> = {
  // Regularisation / complexity
  C: 'Inverse regularisation strength — smaller values apply stronger regularisation and prevent overfitting.',
  alpha: 'Regularisation strength (L1/L2). Higher values add more penalty to large coefficients to reduce overfitting.',
  l1_ratio: 'Mix ratio between L1 and L2 regularisation (ElasticNet). 0 = pure Ridge, 1 = pure Lasso.',
  penalty: 'Type of regularisation to apply (l1 = sparse, l2 = smooth, elasticnet = both, none).',
  reg_alpha: 'L1 regularisation term. Encourages sparsity in the leaf weights (XGBoost / LightGBM).',
  reg_lambda: 'L2 regularisation term. Smooths the leaf weights to reduce overfitting (XGBoost / LightGBM).',
  l2_regularization: 'L2 regularisation added to each leaf (HistGradientBoosting). Higher = more regularised.',
  // Tree structure
  max_depth: 'Maximum depth of each tree. Deeper trees capture more patterns but risk overfitting. -1 means unlimited (LightGBM).',
  n_estimators: 'Number of trees to build. More trees improve accuracy but increase training time.',
  min_samples_split: 'Minimum samples required to split a node. Higher values prevent the tree from learning noise.',
  min_samples_leaf: 'Minimum samples that must remain in a leaf after a split. Acts as smoothing; higher values reduce variance.',
  min_child_weight: 'Minimum sum of instance weights in a child node (XGBoost). Higher values make the model more conservative.',
  min_child_samples: 'Minimum data points in a leaf node (LightGBM). Increase to prevent overfitting on small datasets.',
  max_leaf_nodes: 'Maximum number of leaves in each tree (HistGradientBoosting). Limits model complexity.',
  max_bins: 'Number of bins used for feature discretisation (HistGradientBoosting). More bins → finer splits but slower.',
  num_leaves: 'Maximum number of leaves per tree (LightGBM). Directly controls model complexity; prefer over max_depth.',
  bootstrap: 'Whether to train each tree on a bootstrap sample (random sub-sample with replacement) of the data.',
  criterion: 'Function used to measure split quality (gini, entropy for classifiers; squared_error, absolute_error for regressors).',
  // Boosting / learning
  learning_rate: 'Shrinkage factor applied to each tree contribution. Lower values need more trees but generalise better.',
  subsample: 'Fraction of training samples used for each tree. Introduces randomness to reduce overfitting.',
  colsample_bytree: 'Fraction of features sampled per tree. Reduces correlation between trees and overfitting.',
  gamma: 'Minimum loss reduction required to make a further partition on a leaf node (XGBoost). Higher = more conservative.',
  boosting_type: 'LightGBM boosting algorithm: gbdt (traditional gradient), dart (dropout), goss (gradient-based sampling).',
  // SVM / kernel
  kernel: 'Kernel function used to transform the feature space (rbf, linear, poly, sigmoid). rbf works well for most tasks.',
  // KNN
  n_neighbors: 'Number of nearest neighbours to consider. Lower = more flexible model; higher = smoother decision boundary.',
  weights: 'How neighbours are weighted: uniform (all equal) or distance (closer neighbours have more influence).',
  algorithm: 'Algorithm used to find nearest neighbours: ball_tree, kd_tree, or brute force.',
  // Linear models
  fit_intercept: 'Whether to fit a bias term. Disable only if data is already centred.',
  solver: 'Optimisation algorithm. Choice depends on dataset size and regularisation type.',
  max_iter: 'Maximum number of iterations for the solver to converge. Increase if training shows convergence warnings.',
  selection: 'Coefficient update order for coordinate descent (Lasso/ElasticNet): cyclic (stable) or random (faster).',
  // Naive Bayes
  var_smoothing: 'Portion of the largest variance added to all variances for numerical stability (Gaussian NB).',
};

/**
 * Returns a short explanation for a hyperparameter name such as "max_depth",
 * "learning_rate", "C", "alpha", etc. Returns undefined for unknown params.
 */
export function getHyperparamDescription(param: string): string | undefined {
  return HYPERPARAM_DESCRIPTIONS[param];
}

const TRAINING_CONFIG_DESCRIPTIONS: Record<string, string> = {
  'Target Column': 'The column the model is trained to predict.',
  'CV Enabled': 'Whether cross-validation was used to estimate generalisation performance.',
  'CV Method': 'Cross-validation strategy: KFold splits data into k equal folds; StratifiedKFold preserves class proportions; TimeSeriesSplit respects temporal order.',
  'CV Folds': 'Number of cross-validation folds. More folds → more reliable estimate but longer training time.',
  'CV Shuffle': 'Whether to randomly shuffle samples before splitting into folds. Recommended unless data is time-ordered.',
  'CV Random State': 'Seed for reproducible fold shuffling.',
  'Strategy': 'Search strategy for hyperparameter tuning: random (fast, approximate), grid (exhaustive), halving_grid / halving_random (successive halving — prunes poor candidates early), optuna (Bayesian optimisation).',
  'Strategy Params': 'Additional settings passed to the search strategy (e.g. sampler and pruner for Optuna, factor for halving).',
  'Metric': 'Scoring metric used to rank candidate hyperparameter sets during tuning.',
  'Trials': 'Number of hyperparameter configurations to evaluate. More trials → better search coverage but longer run time.',
};

/** Returns a description for a Training Configuration field label. */
export function getTrainingConfigDescription(field: string): string | undefined {
  return TRAINING_CONFIG_DESCRIPTIONS[field];
}

/** Pretty-print a scoring metric name, e.g. "f1_weighted" → "F1 Weighted". */
export const formatMetricName = (metric?: string | null): string => {
  if (!metric) return '';
  const map: Record<string, string> = {
    accuracy: 'Accuracy', f1: 'F1', precision: 'Precision', recall: 'Recall',
    roc_auc: 'ROC AUC', r2: 'R²', mse: 'MSE', mae: 'MAE', rmse: 'RMSE', mape: 'MAPE',
    f1_weighted: 'F1 Weighted', precision_weighted: 'Precision Weighted',
    recall_weighted: 'Recall Weighted', roc_auc_weighted: 'ROC AUC Weighted',
    f1_macro: 'F1 Macro', f1_micro: 'F1 Micro',
    precision_macro: 'Precision Macro', recall_macro: 'Recall Macro',
    roc_auc_ovr: 'ROC AUC OVR', roc_auc_ovo: 'ROC AUC OVO',
    roc_auc_ovr_weighted: 'ROC AUC OVR Weighted', roc_auc_ovo_weighted: 'ROC AUC OVO Weighted',
    balanced_accuracy: 'Balanced Accuracy', matthews_corrcoef: 'MCC',
    explained_variance: 'Explained Variance', log_loss: 'Log Loss',
    neg_mean_squared_error: 'MSE', neg_mean_absolute_error: 'MAE',
    neg_root_mean_squared_error: 'RMSE', neg_log_loss: 'Log Loss',
  };
  return map[metric] || metric.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

export const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};
