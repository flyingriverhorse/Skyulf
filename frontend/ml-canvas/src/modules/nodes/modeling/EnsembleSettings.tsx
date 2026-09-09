import { useEffect, useState, useMemo, useId } from 'react';
import { Boxes, Info, X } from 'lucide-react';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { registryApi } from '../../../core/api/registry';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { incomingModelNodes, connectedModelPatch } from './ensembleSettings/connectedModels';
import {
  baseOptions, defaultBaseEstimators, defaultFinalEstimator, defaultMetric,
  inferTaskFromColumn, resolveModelId, optionLabelMap, type UpdateFn,
} from './ensembleSettings/modelOptions';
import {
  StrategyOptions, VotingWeightsSection, ParallelJobsSection, CalibrationSection,
  BaseParamsSection, PrimarySetupSection, EnsembleActions,
} from './ensembleSettings/EnsembleFormSections';
import { AdvancedTuningOptions } from './ensembleSettings/AdvancedTuningOptions';
import { CrossValidationSection } from './ensembleSettings/CrossValidationSection';

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
  const { availableColumns, upstreamTarget, datasetId, runJob, isSubmitting, submissionMessage, runFeedback } = useTrainingNodeContext(nodeId);
  const runHelpId = useId();

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
    const incomingModels = incomingModelNodes(nodeId, nodes, edges);
    const patch = connectedModelPatch(incomingModels, config);
    if (patch) onChange({ ...config, ...patch });
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
  const hasWiredModel = useMemo(
    () => incomingModelNodes(nodeId, nodes, edges).length > 0,
    [nodeId, edges, nodes],
  );

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

  const isAdvanced = config.run_mode === 'advanced';
  const tooFewModels = (config.base_estimators?.length ?? 0) < 2;

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800 rounded text-xs text-purple-700 dark:text-purple-300 flex justify-between items-start gap-2">
          <span>Combine several models into one. <strong>Voting</strong> averages their predictions; <strong>Stacking</strong> trains a meta-learner on their out-of-fold predictions.</span>
          <button
            type="button"
            aria-label="Dismiss ensemble information"
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
          <PrimarySetupSection config={config} update={update} currentOptions={currentOptions} columns={availableColumns} tooFewModels={tooFewModels} />

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

      <EnsembleActions config={config} datasetId={datasetId} runJob={runJob} isSubmitting={isSubmitting}
        submissionMessage={submissionMessage} runFeedback={runFeedback} runHelpId={runHelpId} tooFewModels={tooFewModels} />
    </div>
  );
}

// `Boxes` is exported for the node definition's icon to keep imports co-located.
export { Boxes as EnsembleIcon };
