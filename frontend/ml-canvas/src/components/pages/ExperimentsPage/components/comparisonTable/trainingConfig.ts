import type { JobInfo } from '../../../../../core/api/jobs';
import { hasTuningMetadata } from '../../utils/jobMeta';
import type { Config } from './types';

type Scalar = string | number;

/** Empty config scalars use their field fallback, while zero remains a number. */
function scalar(value: unknown, fallback: Scalar = '-'): Scalar {
  return value === undefined || value === null || value === ''
    ? fallback
    : (typeof value === 'number' ? value : String(value));
}

function strategyParams(tuningConfig: Config | undefined): string {
  const params = tuningConfig?.strategy_params as Config | undefined;
  const strategy = String(tuningConfig?.strategy ?? tuningConfig?.search_strategy ?? '');
  if (params && Object.keys(params).length > 0) return JSON.stringify(params);
  if (strategy === 'optuna') return 'sampler: tpe · pruner: median (defaults)';
  if (strategy === 'halving_grid' || strategy === 'halving_random') return 'factor: 3 · min: exhaust (defaults)';
  return '-';
}

const CV_VALUES = new Map<string, (source: Config) => Scalar>([
  ['CV Enabled', source => source.cv_enabled ? 'Yes' : 'No'],
  ['CV Method', source => source.cv_enabled ? scalar(source.cv_type, 'Unknown') : '-'],
  ['CV Folds', source => source.cv_enabled ? scalar(source.cv_folds) : '-'],
  ['CV Shuffle', source => source.cv_enabled ? (source.cv_shuffle ? 'Yes' : 'No') : '-'],
  ['CV Random State', source => source.cv_enabled ? scalar(source.cv_random_state) : '-'],
]);

const TUNING_VALUES = new Map<string, (source: Config | undefined) => Scalar>([
  ['Strategy', source => scalar(source?.strategy ?? source?.search_strategy)],
  ['Strategy Params', strategyParams],
  ['Metric', source => scalar(source?.metric)],
  ['Trials', source => scalar(source?.n_trials)],
]);

/** Tuning runs read CV settings inside tuning_config; basic runs use node settings. */
export function trainingConfigValue(field: string, job: JobInfo, config: Config): Scalar {
  if (field === 'Target Column') return scalar(config.target_column ?? job.target_column);
  const tuningConfig = config.tuning_config as Config | undefined;
  const cvSource = hasTuningMetadata(job) && tuningConfig ? tuningConfig : config;
  const cvValue = CV_VALUES.get(field);
  if (cvValue) return cvValue(cvSource);
  const tuningValue = TUNING_VALUES.get(field);
  return tuningValue ? tuningValue(tuningConfig) : '-';
}
