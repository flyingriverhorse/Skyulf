import { useEffect, useState, useMemo } from 'react';
import {
  Play, Boxes, ChevronRight, BarChart3, AlertTriangle, Info, X, Sparkles,
} from 'lucide-react';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { MultiSelectChips } from './components/MultiSelectChips';
import { BaseModelParamsEditor } from './components/BaseModelParamsEditor';
import { HelpTooltip } from './components/HelpTooltip';
import type { ColumnProfile } from '../../../core/api/client';
import { registryApi } from '../../../core/api/registry';
import { StrategySettingsModal } from './components/StrategySettingsModal';
import { Settings2 } from 'lucide-react';
import { useGraphStore } from '../../../core/store/useGraphStore';

type Task = 'classification' | 'regression';
type Strategy = 'voting' | 'stacking';
type RunMode = 'basic' | 'advanced';

export interface EnsembleConfig {
  task: Task;
  // Set once the user manually toggles Task; suppresses target-dtype auto-detection.
  task_manual?: boolean;
  strategy: Strategy;
  model_type: string;
  base_estimators: string[];
  voting: 'soft' | 'hard';
  final_estimator: string;
  cv: number;
  // Stacking only: feed the original features to the meta-learner alongside
  // the base models' predictions.
  passthrough?: boolean;
  // Voting only: per-base-model relative weights, keyed by base-learner key so
  // they stay aligned when the selection is reordered. Missing key → weight 1.
  weights?: Record<string, number>;
  // Base models fit in parallel. 1 = sequential, -1 = all cores.
  n_jobs?: number;
  // Classification only: wrap each base classifier in CalibratedClassifierCV so
  // its predicted probabilities are well-calibrated (better soft voting/stacking).
  calibrate_base_models?: boolean;
  calibration_method?: 'sigmoid' | 'isotonic';
  calibration_cv?: number;
  target_column: string;
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
  cv_time_column?: string;
  // Per-base-model fixed hyperparameters: { base_key: { param: value } }.
  base_estimator_params?: Record<string, Record<string, unknown>>;
  final_estimator_params?: Record<string, unknown>;
  // Advanced (hyperparameter tuning) mode.
  run_mode: RunMode;
  search_strategy: string;
  n_trials: number;
  metric: string;
  tune_base_models: boolean;
  random_state: number;
  strategy_params?: Record<string, unknown>;
}

type Option = { label: string; value: string };

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

function baseOptions(task: Task, availableIds?: Set<string>, currentSelection: string[] = []): Option[] {
  const base = task === 'classification' ? [...CLF_OPTIONS] : [...REG_OPTIONS];
  if (task === 'classification') {
    if ((availableIds && availableIds.has('xgboost_classifier')) || currentSelection.includes('xgboost')) {
      base.push({ label: 'XGBoost', value: 'xgboost' });
    }
    if ((availableIds && availableIds.has('lgbm_classifier')) || currentSelection.includes('lightgbm')) {
      base.push({ label: 'LightGBM', value: 'lightgbm' });
    }
  } else {
    if ((availableIds && availableIds.has('xgboost_regressor')) || currentSelection.includes('xgboost')) {
      base.push({ label: 'XGBoost', value: 'xgboost' });
    }
    if ((availableIds && availableIds.has('lgbm_regressor')) || currentSelection.includes('lightgbm')) {
      base.push({ label: 'LightGBM', value: 'lightgbm' });
    }
  }
  return base;
}

function resolveModelId(task: Task, strategy: Strategy): string {
  return `${strategy}_${task === 'classification' ? 'classifier' : 'regressor'}`;
}

function defaultBaseEstimators(task: Task): string[] {
  return task === 'classification'
    ? ['random_forest', 'logistic_regression', 'gradient_boosting']
    : ['random_forest', 'gradient_boosting', 'ridge'];
}

function defaultFinalEstimator(task: Task): string {
  return task === 'classification' ? 'logistic_regression' : 'ridge';
}

function defaultMetric(task: Task): string {
  return task === 'classification' ? 'accuracy' : 'r2';
}

/**
 * Infer the ML task from the target column's profile, mirroring the backend EDA
 * heuristic (`skyulf.profiling`): float → regression, boolean / string /
 * categorical → classification, integer → classification only when it is
 * low-cardinality (reads as discrete class labels) otherwise regression.
 * Returns ``null`` when the dtype is unknown so callers keep the current task.
 */
function inferTaskFromColumn(col: ColumnProfile | undefined): Task | null {
  if (!col) return null;
  const dt = String(col.dtype).toLowerCase();
  if (dt.includes('bool')) return 'classification';
  if (dt.includes('object') || dt.includes('string') || dt.includes('category') || dt.includes('text')) {
    return 'classification';
  }
  if (dt.includes('float') || dt.includes('double') || dt.includes('decimal')) {
    return 'regression';
  }
  if (dt.includes('int')) {
    return col.unique_count > 0 && col.unique_count <= 20 ? 'classification' : 'regression';
  }
  return null;
}

function metricOptions(task: Task): Option[] {
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

const SEARCH_STRATEGIES: Option[] = [
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

function lookupTaskFromModelType(modelType: string): 'classification' | 'regression' | null {
  if (modelType in LOOKUP_BASE_KEY_BY_MODEL_TYPE.classification) return 'classification';
  if (modelType in LOOKUP_BASE_KEY_BY_MODEL_TYPE.regression) return 'regression';
  if (modelType.endsWith('_classifier') || modelType === 'logistic_regression' || modelType === 'gaussian_nb' || modelType === 'sgd_classifier' || modelType === 'svc') {
    return 'classification';
  }
  if (modelType.endsWith('_regressor') || modelType === 'linear_regression' || modelType.includes('regression') || modelType === 'svr') {
    return 'regression';
  }
  return null;
}

function resolveEnsembleBaseKey(modelType: string, task: Task): string | null {
  return LOOKUP_BASE_KEY_BY_MODEL_TYPE[task][modelType] ?? null;
}

function optionLabelMap(options: Option[]): Record<string, string> {
  return Object.fromEntries(options.map((o) => [o.value, o.label]));
}

type UpdateFn = (patch: Partial<EnsembleConfig>) => void;

/** Two-option segmented control (Classification/Regression, Voting/Stacking). */
function SegmentedToggle({ options, value, onSelect }: {
  options: Option[];
  value: string;
  onSelect: (v: string) => void;
}) {
  return (
    <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => { onSelect(opt.value); }}
          className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
            value === opt.value
              ? 'bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-300 shadow-sm'
              : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/** Strategy-specific options: voting type (clf voting) or final estimator + CV (stacking). */
function StrategyOptions({ config, update, options }: { config: EnsembleConfig; update: UpdateFn; options: Option[] }) {
  if (config.strategy === 'voting') {
    if (config.task !== 'classification') return null;
    return (
      <div>
        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Voting Type</span>
        <SegmentedToggle
          options={[{ label: 'Soft (probabilities)', value: 'soft' }, { label: 'Hard (majority)', value: 'hard' }]}
          value={config.voting}
          onSelect={(v) => { update({ voting: v as 'soft' | 'hard' }); }}
        />
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div>
        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Final Estimator (meta-learner)</span>
        <select
          value={config.final_estimator}
          onChange={(e) => { update({ final_estimator: e.target.value }); }}
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      <div>
        <div className="flex items-center gap-1 mb-1">
          <span className="block text-xs font-medium text-gray-700 dark:text-gray-300">Stacking CV Folds</span>
          <HelpTooltip text="Out-of-fold folds used to train the final estimator without leakage. Keep small (e.g. 3) when also running an outer hyperparameter search." />
        </div>
        <input
          type="number"
          min={2}
          max={10}
          value={config.cv}
          onChange={(e) => { update({ cv: Number(e.target.value) }); }}
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
        />
      </div>
      <div>
        <label className="flex items-center gap-2 text-xs font-medium text-gray-700 dark:text-gray-300">
          <input
            type="checkbox"
            checked={config.passthrough === true}
            onChange={(e) => { update({ passthrough: e.target.checked }); }}
            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
          />
          Passthrough features
        </label>
        <p className="mt-1 pl-6 text-xs text-gray-500 dark:text-gray-400">
          Let the meta-learner also see the original features, not just the base predictions.
        </p>
      </div>
    </div>
  );
}

/** Voting only: per-base-model relative weights (sklearn `weights=`). */
function VotingWeightsSection({ config, update, optionLabels }: {
  config: EnsembleConfig;
  update: UpdateFn;
  optionLabels: Record<string, string>;
}) {
  const bases = config.base_estimators ?? [];
  if (config.strategy !== 'voting' || bases.length === 0) return null;
  const weights = config.weights ?? {};
  const setWeight = (key: string, value: number) => {
    update({ weights: { ...weights, [key]: value } });
  };
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Weights</span>
        <HelpTooltip text="Relative weight of each base model when averaging votes/probabilities. Leave at 1 for equal weighting; raise a model's weight to trust it more." />
      </div>
      <div className="space-y-1.5">
        {bases.map((key) => (
          <div key={key} className="flex items-center gap-2">
            <span className="flex-1 text-xs text-gray-700 dark:text-gray-300 truncate">{optionLabels[key] ?? key}</span>
            <input
              type="number"
              min={0}
              step={0.5}
              value={weights[key] ?? 1}
              onChange={(e) => { setWeight(key, Number(e.target.value)); }}
              className="w-20 border border-gray-300 dark:border-gray-600 rounded-lg p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Parallel base-model fitting (sklearn `n_jobs`). Applies to all ensembles. */
function ParallelJobsSection({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  const options: Option[] = [
    { label: 'Sequential (1)', value: '1' },
    { label: '2 cores', value: '2' },
    { label: '4 cores', value: '4' },
    { label: '8 cores', value: '8' },
    { label: 'All cores (-1)', value: '-1' },
  ];
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Parallel Jobs</span>
        <HelpTooltip text="How many base models to fit in parallel. -1 uses all CPU cores; 1 trains sequentially." />
      </div>
      <select
        value={String(config.n_jobs ?? 1)}
        onChange={(e) => { update({ n_jobs: Number(e.target.value) }); }}
        className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}

/** Classification only: wrap each base classifier in CalibratedClassifierCV. */
function CalibrationSection({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  if (config.task !== 'classification') return null;
  const enabled = config.calibrate_base_models === true;
  return (
    <div className="space-y-1.5">
      <div>
        <label className="flex items-center gap-2 text-xs font-medium text-gray-700 dark:text-gray-300">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => { update({ calibrate_base_models: e.target.checked }); }}
            className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
          />
          Calibrate base models
        </label>
        <p className="mt-1 pl-6 text-xs text-gray-500 dark:text-gray-400">
          Wrap each base classifier in CalibratedClassifierCV so its probabilities are
          well-calibrated — improves soft voting and stacking.
        </p>
      </div>
      {enabled && (
        <div className="grid grid-cols-2 gap-3 pl-6">
          <div>
            <span className="block text-xs text-gray-500 mb-1">Method</span>
            <select
              value={config.calibration_method ?? 'sigmoid'}
              onChange={(e) => { update({ calibration_method: e.target.value as 'sigmoid' | 'isotonic' }); }}
              className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
            >
              <option value="sigmoid">Sigmoid (Platt)</option>
              <option value="isotonic">Isotonic</option>
            </select>
          </div>
          <div>
            <span className="block text-xs text-gray-500 mb-1">CV Folds</span>
            <input
              type="number"
              min={2}
              max={10}
              value={config.calibration_cv ?? 3}
              onChange={(e) => { update({ calibration_cv: Number(e.target.value) }); }}
              className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
            />
          </div>
        </div>
      )}
    </div>
  );
}

/** Collapsible outer evaluation cross-validation (same shape as the training nodes). */
function CrossValidationSection({ config, update, showCV, setShowCV, columns }: {
  config: EnsembleConfig;
  update: UpdateFn;
  showCV: boolean;
  setShowCV: (v: boolean) => void;
  columns: ColumnProfile[];
}) {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => { setShowCV(!showCV); }}
        className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-purple-500" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">Cross Validation</span>
        </div>
        <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${showCV ? 'rotate-90' : ''}`} />
      </button>
      {showCV && (
        <div className="p-3 space-y-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input
              type="checkbox"
              checked={config.cv_enabled !== false}
              onChange={(e) => { update({ cv_enabled: e.target.checked }); }}
              className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
            />
            Enable Cross-Validation
          </label>
          {config.cv_enabled !== false && (
            <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="block text-xs text-gray-500 mb-1">Folds</span>
                  <input
                    type="number"
                    min={2}
                    value={config.cv_folds ?? 5}
                    onChange={(e) => { update({ cv_folds: Number(e.target.value) }); }}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                  />
                </div>
                <div>
                  <span className="block text-xs text-gray-500 mb-1">Method</span>
                  <select
                    value={config.cv_type ?? 'k_fold'}
                    onChange={(e) => { update({ cv_type: e.target.value }); }}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                  >
                    <option value="k_fold">K-Fold</option>
                    <option value="stratified_k_fold">Stratified</option>
                    <option value="time_series_split">Time Series</option>
                    <option value="shuffle_split">Shuffle Split</option>
                    <option value="nested_cv">Nested CV</option>
                  </select>
                </div>
              </div>

              {config.cv_type === 'time_series_split' && (
                <div className="space-y-2">
                  <div className="flex items-start gap-1.5 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs text-amber-700 dark:text-amber-400">
                    <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                    <span>Data must be sorted by time. Select a date column below or ensure your data is pre-sorted.</span>
                  </div>
                  <div>
                    <span className="block text-xs text-gray-500 mb-1">Time Column (optional)</span>
                    <select
                      value={config.cv_time_column ?? ''}
                      onChange={(e) => { update({ cv_time_column: e.target.value }); }}
                      className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    >
                      <option value="">Auto-detect</option>
                      {columns
                        .filter((col) => {
                          const dt = String(col.dtype).toLowerCase();
                          return dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp');
                        })
                        .map((col) => (
                          <option key={col.name} value={col.name}>{col.name}</option>
                        ))
                      }
                      {columns
                        .filter((col) => {
                          const dt = String(col.dtype).toLowerCase();
                          return !(dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp'));
                        })
                        .map((col) => (
                          <option key={col.name} value={col.name}>{col.name}</option>
                        ))
                      }
                    </select>
                  </div>
                </div>
              )}

              <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                <input
                  type="checkbox"
                  checked={config.cv_shuffle !== false}
                  onChange={(e) => { update({ cv_shuffle: e.target.checked }); }}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                Shuffle Data
              </label>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TargetSelector({ config, update, columns }: {
  config: EnsembleConfig;
  update: UpdateFn;
  columns: ColumnProfile[];
}) {
  return (
    <div>
      <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</span>
      {columns.length === 0 ? (
        <input
          type="text"
          value={config.target_column}
          onChange={(e) => { update({ target_column: e.target.value }); }}
          placeholder="Type the target column name"
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
        />
      ) : (
        <select
          value={config.target_column}
          onChange={(e) => { update({ target_column: e.target.value }); }}
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
        >
          <option value="">Select target…</option>
          {columns.map((col) => (
            <option key={col.name} value={col.name}>{col.name}</option>
          ))}
        </select>
      )}
    </div>
  );
}

/** Tuning controls shown in Advanced mode: search strategy, trial budget, metric. */
function AdvancedTuningOptions({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [showBaseTuningDetails, setShowBaseTuningDetails] = useState(false);
  const showStrategyBtn =
    config.search_strategy === 'halving_grid' ||
    config.search_strategy === 'halving_random' ||
    config.search_strategy === 'optuna';

  return (
    <div className="space-y-3 p-3 rounded-lg border border-purple-100 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/10">
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3.5 h-3.5 text-purple-500" />
        <span className="text-xs font-semibold text-purple-700 dark:text-purple-300">Hyperparameter Tuning</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="flex items-center gap-1 text-xs font-medium text-gray-700 dark:text-gray-300">
              Search Strategy
              <HelpTooltip
                placement="bottom-left"
                text="Searches the ensemble's own params (e.g. voting/cv), not each base model. Enable &quot;Tune base model hyperparameters&quot; below to also search each model's params."
              />
            </span>
            {showStrategyBtn && (
              <button
                type="button"
                onClick={() => { setShowStrategyModal(true); }}
                className="text-blue-600 hover:text-blue-700 dark:text-blue-400 p-0.5 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 transition group flex items-center justify-center"
                title={`${config.search_strategy.replace('_', ' ')} Settings`}
              >
                <Settings2 size={13} className="group-hover:rotate-45 transition-transform duration-300" />
              </button>
            )}
          </div>
          <select
            value={config.search_strategy}
            onChange={(e) => {
              update({
                search_strategy: e.target.value,
                strategy_params: {} // clear params on change to prevent carryover
              });
            }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
          >
            {SEARCH_STRATEGIES.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
        <div>
          <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Trials</span>
          <input
            type="number"
            min={1}
            value={config.n_trials}
            onChange={(e) => { update({ n_trials: Number(e.target.value) }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          />
        </div>
      </div>
      <div>
        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Optimize Metric</span>
        <select
          value={config.metric}
          onChange={(e) => { update({ metric: e.target.value }); }}
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
        >
          {metricOptions(config.task).map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      <label className="flex items-start gap-2 text-xs text-gray-700 dark:text-gray-300">
        <input
          type="checkbox"
          checked={config.tune_base_models !== false}
          onChange={(e) => { update({ tune_base_models: e.target.checked }); }}
          className="mt-0.5 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
        />
        <span className="flex-1">
          <span className="inline-flex items-center gap-1">
            <strong>Tune base model hyperparameters</strong>
            <button
              type="button"
              onClick={() => { setShowBaseTuningDetails(!showBaseTuningDetails); }}
              className="text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
              title={showBaseTuningDetails ? 'Hide details' : 'Show details'}
            >
              <ChevronRight
                className={`w-3 h-3 transition-transform ${showBaseTuningDetails ? 'rotate-90' : ''}`}
              />
            </button>
          </span>
          {showBaseTuningDetails && (
            <span className="block mt-1 text-gray-500 dark:text-gray-400">
              Automatically searches each selected base learner&apos;s default parameter range
              (e.g. <code>random_forest__n_estimators</code>), in addition to the ensemble&apos;s own
              params. Ranges are picked automatically per model — uncheck this to tune only the
              ensemble-level params and keep base models at the fixed values set in Basic mode.
            </span>
          )}
        </span>
      </label>

      {showStrategyModal && (
        <StrategySettingsModal
          isOpen={showStrategyModal}
          onClose={() => { setShowStrategyModal(false); }}
          onSave={(p) => { update({ strategy_params: p as Record<string, unknown> }); }}
          strategy={config.search_strategy}
          initialConfig={config.strategy_params}
          modelKey={config.model_type}
        />
      )}
    </div>
  );
}

/** Collapsible per-base-model fixed hyperparameter editor (Basic mode). */
function BaseParamsSection({ config, update, open, setOpen, options }: {
  config: EnsembleConfig;
  update: UpdateFn;
  open: boolean;
  setOpen: (v: boolean) => void;
  options: Option[];
}) {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => { setOpen(!open); }}
        className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-500" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200">Base Model Hyperparameters</span>
        </div>
        <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="p-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <BaseModelParamsEditor
            task={config.task}
            baseEstimators={config.base_estimators ?? []}
            finalEstimator={config.strategy === 'stacking' ? config.final_estimator : undefined}
            optionLabels={optionLabelMap(options)}
            baseParams={config.base_estimator_params ?? {}}
            finalParams={config.final_estimator_params ?? {}}
            onChange={(baseParams, finalParams) => {
              update({ base_estimator_params: baseParams, final_estimator_params: finalParams });
            }}
          />
        </div>
      )}
    </div>
  );
}

export function EnsembleSettings({ config, onChange, nodeId }: {
  config: EnsembleConfig;
  onChange: (c: EnsembleConfig) => void;
  nodeId?: string;
}) {
  // Wide enough (expanded node view) to split the form into two side-by-side
  // columns instead of one long scroll; the sidebar stays single-column.
  const [containerRef, isWide] = useIsWideContainer(560);
  const [showCV, setShowCV] = useState(false);
  const [showBaseParams, setShowBaseParams] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_ensemble'));
  const { availableColumns, upstreamTarget, runJob } = useTrainingNodeContext(nodeId);

  const [availableModelIds, setAvailableModelIds] = useState<Set<string>>(new Set());

  // Fetch available models from backend registry to check for XGBoost/LightGBM
  useEffect(() => {
    let active = true;
    registryApi.getAllNodes()
      .then((nodes) => {
        if (active) {
          setAvailableModelIds(new Set(nodes.map((n) => n.id)));
        }
      })
      .catch((err) => {
        console.error('Failed to resolve model registry for ensemble settings:', err);
      });
    return () => {
      active = false;
    };
  }, []);

  const currentOptions = useMemo(() => {
    const selected = [
      ...(config.base_estimators ?? []),
      ...(config.final_estimator ? [config.final_estimator] : []),
    ];
    return baseOptions(config.task, availableModelIds, selected);
  }, [config.task, availableModelIds, config.base_estimators, config.final_estimator]);

  const update: UpdateFn = (patch) => { onChange({ ...config, ...patch }); };

  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  // Auto-sync ensemble configuration when model nodes are connected on the canvas
  useEffect(() => {
    if (!nodeId) return;

    // Find direct incoming model nodes from the canvas
    const incomingEdges = edges.filter((e) => e.target === nodeId);
    const incomingModels: any[] = [];

    for (const edge of incomingEdges) {
      const src = nodes.find((n) => n.id === edge.source);
      if (src && ['training', 'classification', 'regression', 'text_classification'].includes(src.data.definitionType as string)) {
        incomingModels.push(src);
      }
    }

    if (incomingModels.length === 0) return;

    // 1. Detect Task (classification / regression)
    let detectedTask: Task | null = null;
    for (const m of incomingModels) {
      const mt = m.data.model_type as string;
      if (mt) {
        const t = lookupTaskFromModelType(mt);
        if (t) {
          detectedTask = t;
          break;
        }
      }
    }

    // 2. Identify the Run Mode, search strategy, trial count, and metric
    let detectedRunMode: RunMode = 'basic';
    let detectedSearchStrategy = config.search_strategy;
    let detectedTrials = config.n_trials;
    let detectedMetric = config.metric;

    const advancedNode = incomingModels.find((m) => m.data.run_mode === 'advanced');

    if (advancedNode) {
      detectedRunMode = 'advanced';
      if (advancedNode.data.search_strategy) {
        detectedSearchStrategy = advancedNode.data.search_strategy;
      }
      if (advancedNode.data.n_trials) {
        detectedTrials = advancedNode.data.n_trials;
      }
      if (advancedNode.data.metric) {
        detectedMetric = advancedNode.data.metric;
      }
    }

    // 3. Resolve base estimators & parameter maps from all connected nodes
    // Honour a manual task choice: once the user picks a task by hand
    // (`task_manual`), wiring a model must not force it back.
    const taskToUse = config.task_manual ? config.task : (detectedTask || config.task);
    const resolvedBaseEstimators: string[] = [];
    const resolvedBaseParams: Record<string, Record<string, unknown>> = {
      ...(config.base_estimator_params || {}),
    };

    for (const m of incomingModels) {
      const mt = m.data.model_type as string;
      if (mt) {
        const baseKey = resolveEnsembleBaseKey(mt, taskToUse);
        if (baseKey) {
          if (!resolvedBaseEstimators.includes(baseKey)) {
            resolvedBaseEstimators.push(baseKey);
          }
          // Merge hyperparameters to base params so they pre-fill the form
          const hp = m.data.hyperparameters || m.data.search_space || m.data.params;
          if (hp && typeof hp === 'object' && Object.keys(hp as object).length > 0) {
            resolvedBaseParams[baseKey] = {
              ...(resolvedBaseParams[baseKey] || {}),
              ...(hp as Record<string, unknown>),
            };
          }
        }
      }
    }

    // 4. Trace target column and cross-validation from the first incoming model
    const firstModel = incomingModels[0];
    const detectedTarget = firstModel.data.target_column || config.target_column;
    const detectedCVEnabled = firstModel.data.cv_enabled !== undefined ? firstModel.data.cv_enabled : config.cv_enabled;
    const detectedCVFolds = firstModel.data.cv_folds || config.cv_folds;
    const detectedCVType = firstModel.data.cv_type || config.cv_type;
    const detectedCVShuffle = firstModel.data.cv_shuffle !== undefined ? firstModel.data.cv_shuffle : config.cv_shuffle;
    const detectedCVRandomState = firstModel.data.cv_random_state || config.cv_random_state;
    const detectedCVTimeColumn = firstModel.data.cv_time_column || config.cv_time_column;

    // 5. Build state comparison patch and fire onChange conditionally
    const patch: Partial<EnsembleConfig> = {};

    if (detectedTask && detectedTask !== config.task && !config.task_manual) {
      patch.task = detectedTask;
      patch.model_type = resolveModelId(detectedTask, config.strategy);
    }

    if (detectedRunMode !== config.run_mode) {
      patch.run_mode = detectedRunMode;
    }

    if (detectedSearchStrategy !== config.search_strategy) {
      patch.search_strategy = detectedSearchStrategy;
    }

    if (detectedTrials !== config.n_trials) {
      patch.n_trials = detectedTrials;
    }

    if (detectedMetric !== config.metric) {
      patch.metric = detectedMetric;
    }

    const isBaseDiff =
      resolvedBaseEstimators.length > 0 &&
      (resolvedBaseEstimators.length !== config.base_estimators?.length ||
        !resolvedBaseEstimators.every((v) => config.base_estimators?.includes(v)));

    if (isBaseDiff) {
      patch.base_estimators = resolvedBaseEstimators;
    }

    if (detectedTarget && detectedTarget !== config.target_column) {
      patch.target_column = detectedTarget;
    }

    if (detectedCVEnabled !== config.cv_enabled) {
      patch.cv_enabled = detectedCVEnabled;
    }

    if (detectedCVFolds !== config.cv_folds) {
      patch.cv_folds = detectedCVFolds;
    }

    if (detectedCVType !== config.cv_type) {
      patch.cv_type = detectedCVType;
    }

    if (detectedCVShuffle !== config.cv_shuffle) {
      patch.cv_shuffle = detectedCVShuffle;
    }

    if (detectedCVRandomState !== config.cv_random_state) {
      patch.cv_random_state = detectedCVRandomState;
    }

    if (detectedCVTimeColumn !== config.cv_time_column) {
      patch.cv_time_column = detectedCVTimeColumn;
    }

    if (Object.keys(patch).length > 0) {
      onChange({
        ...config,
        ...patch,
        base_estimator_params: resolvedBaseParams,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    nodeId,
    edges,
    nodes,
    config.task,
    config.strategy,
    config.run_mode,
    config.search_strategy,
    config.n_trials,
    config.metric,
    config.base_estimators,
    config.target_column,
    config.cv_enabled,
    config.cv_folds,
    config.cv_type,
    config.cv_shuffle,
    config.cv_random_state,
    config.cv_time_column,
    config.base_estimator_params,
    onChange,
  ]);

  // Auto-fill the target from an upstream Feature/Target split (once, when empty).
  useEffect(() => {
    if (upstreamTarget && !config.target_column) {
      onChange({ ...config, target_column: upstreamTarget });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamTarget, config.target_column]);

  // Are any model nodes wired into this ensemble? When so, the auto-sync effect
  // above drives the task from those nodes, so the target-dtype inference below
  // stands down to avoid the two paths fighting over `config.task`.
  const hasWiredModel = useMemo(() => {
    if (!nodeId) return false;
    return edges.some((e) => {
      if (e.target !== nodeId) return false;
      const src = nodes.find((n) => n.id === e.source);
      return (
        !!src &&
        ['training', 'classification', 'regression', 'text_classification'].includes(
          src.data.definitionType as string,
        )
      );
    });
  }, [nodeId, edges, nodes]);

  // Auto-detect the task from the chosen target column's dtype/cardinality, the
  // same way the EDA profiler infers it, so the four ensemble ids stay aligned
  // with the target without the user toggling Task by hand. Only flips when the
  // inference actually disagrees with the current task, and stands down once the
  // user has manually picked a task (`task_manual`).
  useEffect(() => {
    if (hasWiredModel || config.task_manual || !config.target_column) return;
    const col = availableColumns.find((c) => c.name === config.target_column);
    const inferred = inferTaskFromColumn(col);
    if (inferred && inferred !== config.task) {
      onChange({
        ...config,
        task: inferred,
        model_type: resolveModelId(inferred, config.strategy),
        base_estimators: defaultBaseEstimators(inferred),
        final_estimator: defaultFinalEstimator(inferred),
        metric: defaultMetric(inferred),
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasWiredModel, config.task_manual, config.target_column, config.strategy, availableColumns]);

  const onTask = (task: Task) => {
    update({
      task,
      // Manual choice locks the task so auto-detection stops overriding it.
      task_manual: true,
      model_type: resolveModelId(task, config.strategy),
      base_estimators: defaultBaseEstimators(task),
      final_estimator: defaultFinalEstimator(task),
      metric: defaultMetric(task),
    });
  };

  const onStrategy = (strategy: Strategy) => {
    update({ strategy, model_type: resolveModelId(config.task, strategy) });
  };

  const isAdvanced = config.run_mode === 'advanced';
  const tooFewModels = (config.base_estimators?.length ?? 0) < 2;

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800 rounded text-xs text-purple-700 dark:text-purple-300 flex justify-between items-start gap-2">
          <span>Combine several models into one. <strong>Voting</strong> averages their predictions; <strong>Stacking</strong> trains a meta-learner on their out-of-fold predictions.</span>
          <button
            type="button"
            onClick={() => { setShowInfo(false); sessionStorage.setItem('hide_info_ensemble', 'true'); }}
            className="text-purple-400 hover:text-purple-600 dark:hover:text-purple-200"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-1 pb-4">
        <div className={isWide ? 'grid grid-cols-2 gap-x-6 gap-y-5 items-start' : 'space-y-5'}>
          {/* Left column: primary setup */}
          <div className="space-y-5">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Task</span>
            {config.task_manual ? (
              <button
                type="button"
                onClick={() => { update({ task_manual: false }); }}
                className="text-[10px] text-purple-600 dark:text-purple-400 hover:underline flex items-center gap-0.5"
                title="Re-enable automatic detection from the target column"
              >
                <Sparkles className="w-2.5 h-2.5" /> Auto-detect
              </button>
            ) : (
              <span className="text-[10px] text-gray-400 dark:text-gray-500 flex items-center gap-0.5" title="Inferred from the target column's type">
                <Sparkles className="w-2.5 h-2.5" /> Auto
              </span>
            )}
          </div>
          <SegmentedToggle
            options={[{ label: 'Classification', value: 'classification' }, { label: 'Regression', value: 'regression' }]}
            value={config.task}
            onSelect={(v) => { onTask(v as Task); }}
          />
        </div>

        <div className="space-y-1.5">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Strategy</span>
          <SegmentedToggle
            options={[{ label: 'Voting', value: 'voting' }, { label: 'Stacking', value: 'stacking' }]}
            value={config.strategy}
            onSelect={(v) => { onStrategy(v as Strategy); }}
          />
        </div>

        <div className="space-y-1.5">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Training Mode</span>
          <SegmentedToggle
            options={[{ label: 'Basic', value: 'basic' }, { label: 'Advanced (Tuning)', value: 'advanced' }]}
            value={config.run_mode}
            onSelect={(v) => { update({ run_mode: v as RunMode }); }}
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Base Models</span>
            <span className="text-[10px] text-gray-400">{config.base_estimators?.length ?? 0} selected</span>
          </div>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Click to add or remove each model in the ensemble.</p>
          <MultiSelectChips
            options={currentOptions}
            selected={config.base_estimators ?? []}
            onChange={(vals) => { update({ base_estimators: vals as string[] }); }}
          />
          <p className="text-[10px] text-gray-400 dark:text-gray-500">
            Tip: wire model nodes into this ensemble&apos;s input to use them as base
            learners automatically — connected models override the selection above.
          </p>
          {tooFewModels && (
            <div className="flex items-start gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>Pick at least two base models for a meaningful ensemble.</span>
            </div>
          )}
        </div>

        <TargetSelector config={config} update={update} columns={availableColumns} />
          </div>

          {/* Right column: strategy options, tuning, calibration, CV */}
          <div className="space-y-5">
            <StrategyOptions config={config} update={update} options={currentOptions} />

        <VotingWeightsSection config={config} update={update} optionLabels={optionLabelMap(currentOptions)} />

        <ParallelJobsSection config={config} update={update} />
        <CalibrationSection config={config} update={update} />

        {isAdvanced && <AdvancedTuningOptions config={config} update={update} />}

        <div className="flex items-start gap-1.5 text-[10px] text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded px-2 py-1.5">
          <Info className="w-3 h-3 mt-0.5 shrink-0" />
          <span>Models like SVC/KNN/Logistic Regression benefit from a <strong>Scaler</strong> node upstream.</span>
        </div>

        {!isAdvanced && (
          <BaseParamsSection config={config} update={update} open={showBaseParams} setOpen={setShowBaseParams} options={currentOptions} />
        )}

        <CrossValidationSection config={config} update={update} showCV={showCV} setShowCV={setShowCV} columns={availableColumns} />
        </div>
        </div>
      </div>

      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          type="button"
          onClick={() => { void runJob(isAdvanced ? 'tuning' : 'training', 'ensemble'); }}
          disabled={!config.target_column || tooFewModels}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          {isAdvanced ? <Sparkles className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
          <span className="text-sm font-semibold">
            {isAdvanced ? 'Start Ensemble Modeling' : 'Start Ensemble Training'}
          </span>
        </button>
      </div>
    </div>
  );
}

// `Boxes` is exported for the node definition's icon to keep imports co-located.
export { Boxes as EnsembleIcon };
