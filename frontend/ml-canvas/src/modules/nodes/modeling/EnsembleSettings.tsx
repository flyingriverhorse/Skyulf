import { useEffect, useState } from 'react';
import {
  Play, Boxes, ChevronRight, BarChart3, AlertTriangle, Info, X, Sparkles,
} from 'lucide-react';
import { useElementSize } from '../../../core/hooks/useElementSize';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { MultiSelectChips } from './components/MultiSelectChips';
import { BaseModelParamsEditor } from './components/BaseModelParamsEditor';
import { HelpTooltip } from './components/HelpTooltip';
import type { ColumnProfile } from '../../../core/api/client';

type Task = 'classification' | 'regression';
type Strategy = 'voting' | 'stacking';
type RunMode = 'basic' | 'advanced';

export interface EnsembleConfig {
  task: Task;
  strategy: Strategy;
  model_type: string;
  base_estimators: string[];
  voting: 'soft' | 'hard';
  final_estimator: string;
  cv: number;
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
}

type Option = { label: string; value: string };

// Base learners mirrored from skyulf.modeling.ensemble (BASE_ESTIMATORS_*).
const CLF_OPTIONS: Option[] = [
  { label: 'Logistic Regression', value: 'logistic_regression' },
  { label: 'Random Forest', value: 'random_forest' },
  { label: 'Gradient Boosting', value: 'gradient_boosting' },
  { label: 'Decision Tree', value: 'decision_tree' },
  { label: 'Gaussian Naive Bayes', value: 'gaussian_nb' },
  { label: 'Support Vector Classifier', value: 'svc' },
  { label: 'K-Nearest Neighbors', value: 'knn' },
];

const REG_OPTIONS: Option[] = [
  { label: 'Linear Regression', value: 'linear_regression' },
  { label: 'Ridge', value: 'ridge' },
  { label: 'Random Forest', value: 'random_forest' },
  { label: 'Gradient Boosting', value: 'gradient_boosting' },
  { label: 'Decision Tree', value: 'decision_tree' },
  { label: 'Support Vector Regressor', value: 'svr' },
  { label: 'K-Nearest Neighbors', value: 'knn' },
];

function baseOptions(task: Task): Option[] {
  return task === 'classification' ? CLF_OPTIONS : REG_OPTIONS;
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

function optionLabelMap(task: Task): Record<string, string> {
  return Object.fromEntries(baseOptions(task).map((o) => [o.value, o.label]));
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
function StrategyOptions({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
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
          {baseOptions(config.task).map((opt) => (
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
  return (
    <div className="space-y-3 p-3 rounded-lg border border-purple-100 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/10">
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3.5 h-3.5 text-purple-500" />
        <span className="text-xs font-semibold text-purple-700 dark:text-purple-300">Hyperparameter Tuning</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Search Strategy</span>
          <select
            value={config.search_strategy}
            onChange={(e) => { update({ search_strategy: e.target.value }); }}
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
        <span>
          <strong>Tune base model hyperparameters</strong> — searches each base learner&apos;s params
          (e.g. <code>random_forest__n_estimators</code>) in addition to the ensemble&apos;s own.
        </span>
      </label>
    </div>
  );
}

/** Collapsible per-base-model fixed hyperparameter editor (Basic mode). */
function BaseParamsSection({ config, update, open, setOpen }: {
  config: EnsembleConfig;
  update: UpdateFn;
  open: boolean;
  setOpen: (v: boolean) => void;
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
            optionLabels={optionLabelMap(config.task)}
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
  const [containerRef] = useElementSize();
  const [showCV, setShowCV] = useState(false);
  const [showBaseParams, setShowBaseParams] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_ensemble'));
  const { availableColumns, upstreamTarget, runJob } = useTrainingNodeContext(nodeId);

  const update: UpdateFn = (patch) => { onChange({ ...config, ...patch }); };

  // Auto-fill the target from an upstream Feature/Target split (once, when empty).
  useEffect(() => {
    if (upstreamTarget && !config.target_column) {
      onChange({ ...config, target_column: upstreamTarget });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamTarget, config.target_column]);

  const onTask = (task: Task) => {
    update({
      task,
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

      <div className="flex-1 overflow-y-auto px-1 pb-4 space-y-5">
        <div className="space-y-1.5">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Task</span>
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
            options={baseOptions(config.task)}
            selected={config.base_estimators ?? []}
            onChange={(vals) => { update({ base_estimators: vals as string[] }); }}
          />
          {tooFewModels && (
            <div className="flex items-start gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>Pick at least two base models for a meaningful ensemble.</span>
            </div>
          )}
        </div>

        <StrategyOptions config={config} update={update} />

        {isAdvanced && <AdvancedTuningOptions config={config} update={update} />}

        <div className="flex items-start gap-1.5 text-[10px] text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded px-2 py-1.5">
          <Info className="w-3 h-3 mt-0.5 shrink-0" />
          <span>Models like SVC/KNN/Logistic Regression benefit from a <strong>Scaler</strong> node upstream.</span>
        </div>

        <TargetSelector config={config} update={update} columns={availableColumns} />

        {!isAdvanced && (
          <BaseParamsSection config={config} update={update} open={showBaseParams} setOpen={setShowBaseParams} />
        )}

        <CrossValidationSection config={config} update={update} showCV={showCV} setShowCV={setShowCV} columns={availableColumns} />
      </div>

      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          type="button"
          onClick={() => { void runJob(isAdvanced ? 'advanced_tuning' : 'basic_training'); }}
          disabled={!config.target_column || tooFewModels}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          {isAdvanced ? <Sparkles className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
          <span className="text-sm font-semibold">
            {isAdvanced ? 'Start Ensemble Tuning' : 'Start Ensemble Training'}
          </span>
        </button>
      </div>
    </div>
  );
}

// `Boxes` is exported for the node definition's icon to keep imports co-located.
export { Boxes as EnsembleIcon };
