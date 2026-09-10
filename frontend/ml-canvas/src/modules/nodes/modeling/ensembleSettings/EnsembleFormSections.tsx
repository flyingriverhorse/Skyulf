import { RunFeedback } from '../../../../components/shared/RunFeedback';
import { TrainingActionFooter } from '../../../../components/shared/TrainingActionFooter';
import type { useTrainingNodeContext } from '../../../../core/hooks/useTrainingNodeContext';
import { useId } from 'react';
import { AlertTriangle, ChevronRight, Sparkles, Loader2, Play } from 'lucide-react';
import type { ColumnProfile } from '../../../../core/api/client';
import { ValidationField } from '../../../../components/shared/ValidationField';
import type { EnsembleConfig } from '../EnsembleSettings';
import { BaseModelParamsEditor } from '../components/BaseModelParamsEditor';
import { MultiSelectChips } from '../components/MultiSelectChips';
import { HelpTooltip } from '../components/HelpTooltip';
import {
  defaultBaseEstimators, defaultFinalEstimator, defaultMetric, resolveModelId, optionLabelMap,
  type Option, type Task, type Strategy, type RunMode, type UpdateFn,
} from './modelOptions';

/** Two-option segmented control (Classification/Regression, Voting/Stacking). */
function SegmentedToggle({ label, options, value, onSelect }: {
  label: string;
  options: Option[];
  value: string;
  onSelect: (v: string) => void;
}) {
  return (
    <div role="group" aria-label={label} className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          aria-pressed={value === opt.value}
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
export function StrategyOptions({ config, update, options }: { config: EnsembleConfig; update: UpdateFn; options: Option[] }) {
  const fieldId = useId();
  if (config.strategy === 'voting') {
    if (config.task !== 'classification') return null;
    return (
      <div>
        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Voting Type</span>
        <SegmentedToggle
          options={[{ label: 'Soft (probabilities)', value: 'soft' }, { label: 'Hard (majority)', value: 'hard' }]}
          label="Voting Type"
          value={config.voting}
          onSelect={(v) => { update({ voting: v as 'soft' | 'hard' }); }}
        />
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div>
        <label htmlFor={`${fieldId}-final_estimator`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Final Estimator (meta-learner)</label>
        <select
          id={`${fieldId}-final_estimator`}
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
          <label htmlFor={`${fieldId}-cv`} className="block text-xs font-medium text-gray-700 dark:text-gray-300">Stacking CV Folds</label>
          <HelpTooltip text="Out-of-fold folds used to train the final estimator without leakage. Keep small (e.g. 3) when also running an outer hyperparameter search." />
        </div>
        <input
          id={`${fieldId}-cv`}
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
export function VotingWeightsSection({ config, update, optionLabels }: {
  config: EnsembleConfig;
  update: UpdateFn;
  optionLabels: Record<string, string>;
}) {
  const fieldId = useId();
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
            <label htmlFor={`${fieldId}-weight-${key}`} className="flex-1 text-xs text-gray-700 dark:text-gray-300 truncate">{optionLabels[key] ?? key}</label>
            <input
              type="number"
              min={0}
              step={0.5}
              id={`${fieldId}-weight-${key}`}
              aria-label={`${optionLabels[key] ?? key} weight`}
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
export function ParallelJobsSection({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  const fieldId = useId();
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
        <label htmlFor={`${fieldId}-n_jobs`} className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Parallel Jobs</label>
        <HelpTooltip text="How many base models to fit in parallel. -1 uses all CPU cores; 1 trains sequentially." />
      </div>
      <select
        id={`${fieldId}-n_jobs`}
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
export function CalibrationSection({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  const fieldId = useId();
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
            <label htmlFor={`${fieldId}-calibration_method`} className="block text-xs text-gray-500 mb-1">Method</label>
            <select
              id={`${fieldId}-calibration_method`}
              aria-label="Calibration Method"
              value={config.calibration_method ?? 'sigmoid'}
              onChange={(e) => { update({ calibration_method: e.target.value as 'sigmoid' | 'isotonic' }); }}
              className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
            >
              <option value="sigmoid">Sigmoid (Platt)</option>
              <option value="isotonic">Isotonic</option>
            </select>
          </div>
          <div>
            <label htmlFor={`${fieldId}-calibration_cv`} className="block text-xs text-gray-500 mb-1">CV Folds</label>
            <input
              id={`${fieldId}-calibration_cv`}
              aria-label="Calibration CV Folds"
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

function TargetSelector({ config, update, columns }: {
  config: EnsembleConfig;
  update: UpdateFn;
  columns: ColumnProfile[];
}) {
  const fieldId = useId();
  return (
    <ValidationField field="target_column">
      <label htmlFor={`${fieldId}-target_column`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</label>
      {columns.length === 0 ? (
        <input
          id={`${fieldId}-target_column`}
          type="text"
          value={config.target_column}
          onChange={(e) => { update({ target_column: e.target.value }); }}
          placeholder="Type the target column name"
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
        />
      ) : (
        <select
          id={`${fieldId}-target_column`}
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
    </ValidationField>
  );
}

/** Collapsible per-base-model fixed hyperparameter editor (Basic mode). */
export function BaseParamsSection({ config, update, open, setOpen, options }: {
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
        aria-expanded={open}
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

/** Task, strategy, model selection and target edits form the primary setup column. */
export function PrimarySetupSection({ config, update, currentOptions, columns, tooFewModels }: {
  config: EnsembleConfig; update: UpdateFn; currentOptions: Option[]; columns: ColumnProfile[]; tooFewModels: boolean;
}) {
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

  return (
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
            <span className="text-[10px] text-gray-500 dark:text-gray-400 flex items-center gap-0.5" title="Inferred from the target column's type">
              <Sparkles className="w-2.5 h-2.5" /> Auto
            </span>
          )}
        </div>
        <SegmentedToggle
          options={[{ label: 'Classification', value: 'classification' }, { label: 'Regression', value: 'regression' }]}
          label="Task"
          value={config.task}
          onSelect={(v) => { onTask(v as Task); }}
        />
      </div>

      <div className="space-y-1.5">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Strategy</span>
        <SegmentedToggle
          options={[{ label: 'Voting', value: 'voting' }, { label: 'Stacking', value: 'stacking' }]}
          label="Strategy"
          value={config.strategy}
          onSelect={(v) => { onStrategy(v as Strategy); }}
        />
      </div>

      <div className="space-y-1.5">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Training Mode</span>
        <SegmentedToggle
          options={[{ label: 'Basic', value: 'basic' }, { label: 'Advanced (Tuning)', value: 'advanced' }]}
          label="Training Mode"
          value={config.run_mode}
          onSelect={(v) => { update({ run_mode: v as RunMode }); }}
        />
      </div>

      <ValidationField field="base_estimators">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Base Models</span>
            <span className="text-[10px] text-gray-400">{config.base_estimators?.length ?? 0} selected</span>
          </div>
          <p className="text-[11px] text-gray-500 dark:text-gray-400">Click to add or remove each model in the ensemble.</p>
          <MultiSelectChips
            ariaLabel="Base Models"
            options={currentOptions}
            selected={config.base_estimators ?? []}
            onChange={(vals) => { update({ base_estimators: vals as string[] }); }}
          />
          <p className="text-[10px] text-gray-500 dark:text-gray-400">
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
      </ValidationField>

      <TargetSelector config={config} update={update} columns={columns} />
    </div>
  );
}

/** Submission prerequisites and their explanation share the same ordered decisions. */
function runAvailability(config: EnsembleConfig, datasetId: string | undefined, tooFewModels: boolean) {
  if (!datasetId) return { canRun: false, description: 'Connect a dataset node upstream and select a dataset to enable this action.' };
  if (!config.target_column?.trim()) return { canRun: false, description: 'Choose a target column to enable this action.' };
  if (tooFewModels) return { canRun: false, description: 'Choose at least two base models to enable this action.' };
  return { canRun: true, description: `${config.run_mode === 'advanced' ? 'Tunes' : 'Trains'} the ${config.strategy} ensemble with ${config.base_estimators.length} base models in the background.` };
}

/** Training action and its persistent receipt use the same parent-owned job context. */
export function EnsembleActions({ config, datasetId, runJob, isSubmitting, submissionMessage, runFeedback, runHelpId, tooFewModels }:
  Pick<ReturnType<typeof useTrainingNodeContext>, 'datasetId' | 'runJob' | 'isSubmitting' | 'submissionMessage' | 'runFeedback'> & {
    config: EnsembleConfig; runHelpId: string; tooFewModels: boolean;
  }) {
  const isAdvanced = config.run_mode === 'advanced';
  const availability = runAvailability(config, datasetId, tooFewModels);
  return (
    <TrainingActionFooter details={
      <p id={runHelpId} className="text-xs text-center text-muted-foreground">
        {availability.description}
      </p>
    }>
      <button
        type="button"
        onClick={() => { void runJob(isAdvanced ? 'tuning' : 'training', 'ensemble'); }}
        disabled={!availability.canRun || isSubmitting}
        aria-describedby={runHelpId}
        className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 action-primary rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 focus-ring"
      >
        {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : isAdvanced ? <Sparkles className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
        <span className="text-sm font-semibold">
          {isSubmitting ? 'Submitting job...' : isAdvanced ? 'Tune ensemble' : 'Train ensemble'}
        </span>
      </button>
      {submissionMessage && <p role="status" aria-atomic="true" className="text-xs text-center text-muted-foreground break-words">{submissionMessage}</p>}
      {runFeedback && <RunFeedback run={runFeedback} task="ensemble" />}
    </TrainingActionFooter>
  );
}
