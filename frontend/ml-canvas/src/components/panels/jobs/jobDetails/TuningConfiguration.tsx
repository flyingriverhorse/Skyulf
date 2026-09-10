import { JobInfo } from '../../../../core/api/jobs';
import { extractEnsembleSummary, formatBaseEstimator, type EnsembleSummary } from '../../../../core/utils/format';

// Tuning config is a free-form server payload that varies by strategy.
type TuningConfig = {
  strategy?: string;
  search_strategy?: string;
  strategy_params?: Record<string, unknown>;
  metric?: string;
  n_trials?: number;
  cv_enabled?: boolean;
  cv_type?: string;
  cv_folds?: number;
  cv_shuffle?: boolean;
};

function TuningEnsemble({ ensemble }: { ensemble: EnsembleSummary | null }) {
  return (
    <>
      {ensemble && (
        <>
          <div className="col-span-2">
            <span className="text-gray-500">Base Models:</span>
            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
              {ensemble.baseEstimators.length > 0
                ? ensemble.baseEstimators.map(formatBaseEstimator).join(', ')
                : '-'}
            </span>
          </div>
          <StackingEnsembleDetails ensemble={ensemble} />
          <VotingEnsembleDetails ensemble={ensemble} />
          {typeof ensemble.nJobs === 'number' && (
            <div>
              <span className="text-gray-500">Parallel Jobs:</span>
              <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{ensemble.nJobs === -1 ? 'All cores' : ensemble.nJobs}</span>
            </div>
          )}
          {ensemble.calibrateBaseModels && (
            <div>
              <span className="text-gray-500">Calibration:</span>
              <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{ensemble.calibrationMethod === 'isotonic' ? 'Isotonic' : 'Sigmoid'}</span>
            </div>
          )}
        </>
      )}
    </>
  );
}

export function TuningConfiguration({ job }: { job: JobInfo }) {
  const config = getTuningConfig(job);
  if (!config) return <div className="text-gray-400 col-span-2">No configuration found</div>;

  const activeStrategy = config.strategy || config.search_strategy || '';
  const strategyParamsDisplay = getStrategyParamsDisplay(config, activeStrategy);

  // For ensemble nodes the structural selection (which base learners
  // were combined) lives alongside the tuning config and is more
  // informative than the bare strategy, so surface it up top.
  const ensemble = extractEnsembleSummary(
    job.model_type,
    config as unknown as Record<string, unknown>,
  );

  return (
    <>
      <TuningEnsemble ensemble={ensemble} />
      <div>
        <span className="text-gray-500">Strategy:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300 capitalize">{activeStrategy || '-'}</span>

      </div>
      <div className="col-span-2">
        <span className="text-gray-500">Strategy Params:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{strategyParamsDisplay}</span>
      </div>
      <div>
        <span className="text-gray-500">Metric:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.metric || '-'}</span>
      </div>
      <div>
        <span className="text-gray-500">Trials:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.n_trials || '-'}</span>
      </div>
      <TuningCrossValidation config={config} />
    </>
  );
}

function StackingEnsembleDetails({ ensemble }: { ensemble: EnsembleSummary }) {
  return (
    <>
      {ensemble.isStacking && (
        <div>
          <span className="text-gray-500">Final Estimator:</span>
          <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
            {ensemble.finalEstimator ? formatBaseEstimator(ensemble.finalEstimator) : '-'}
          </span>
        </div>
      )}
      {ensemble.isStacking && (
        <div>
          <span className="text-gray-500">Passthrough:</span>
          <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{ensemble.passthrough ? 'Yes' : 'No'}</span>
        </div>
      )}
    </>
  );
}

function VotingEnsembleDetails({ ensemble }: { ensemble: EnsembleSummary }) {
  return (
    <>
      {!ensemble.isStacking && ensemble.voting && (
        <div>
          <span className="text-gray-500">Voting:</span>
          <span className="ml-2 font-mono text-gray-700 dark:text-gray-300 capitalize">{ensemble.voting}</span>
        </div>
      )}
      {!ensemble.isStacking && ensemble.weights && ensemble.weights.length === ensemble.baseEstimators.length && (
        <div className="col-span-2">
          <span className="text-gray-500">Model Weights:</span>
          <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">
            {ensemble.baseEstimators.map((m, i) => `${formatBaseEstimator(m)}: ${ensemble.weights?.[i]}`).join(', ')}
          </span>
        </div>
      )}
    </>
  );
}

function getStrategyParamsDisplay(config: TuningConfig, activeStrategy: string) {
  const sp = config.strategy_params;
  const hasStrategyParams = sp && Object.keys(sp).length > 0;
  const strategyParamsDisplay = hasStrategyParams
    ? Object.entries(sp!).map(([k, v]) => `${k}: ${v}`).join(' · ')
    : activeStrategy === 'optuna'
      ? 'sampler: tpe · pruner: median (defaults)'
      : activeStrategy === 'halving_grid' || activeStrategy === 'halving_random'
        ? 'factor: 3 · resource: n_samples · min_resources: exhaust (defaults)'
        : '-';

  return strategyParamsDisplay;
}

function getTuningConfig(job: JobInfo): TuningConfig | undefined {
  const node = (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id);
  const jobConfig = job.config as { tuning_config?: TuningConfig } | undefined;
  const saved = jobConfig?.tuning_config;
  return Object.keys(saved || {}).length > 0 ? saved : node?.params?.tuning_config as TuningConfig | undefined;
}

function TuningCrossValidation({ config }: { config: TuningConfig }) {
  return (
    <>
      <div>
        <span className="text-gray-500">CV Enabled:</span>
        <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_enabled ? 'Yes' : 'No'}</span>
      </div>
      {config.cv_enabled && (
        <>
          <div>
            <span className="text-gray-500">CV Method:</span>
            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_type || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-gray-500">Folds:</span>
            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_folds}</span>
          </div>
          <div>
            <span className="text-gray-500">Shuffle:</span>
            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_shuffle ? 'Yes' : 'No'}</span>
          </div>
        </>
      )}
    </>
  );
}
