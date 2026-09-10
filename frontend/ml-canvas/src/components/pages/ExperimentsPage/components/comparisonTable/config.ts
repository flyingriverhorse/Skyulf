import type { JobInfo } from '../../../../../core/api/jobs';
import { hasTuningMetadata } from '../../utils/jobMeta';
import type { Config, GraphNode } from './types';

// Basic jobs store node settings around their model parameters; tuned jobs
// store best_params directly.
export const getModelParams = (job: { job_type?: string; hyperparameters?: unknown; search_strategy?: string }): Record<string, unknown> => {
  const hp = job.hyperparameters as Record<string, unknown> | undefined;
  if (!hp) return {};
  if (hasTuningMetadata(job)) {
    // For advanced tuning, hyperparameters IS the best_params (or search_space) directly
    return hp;
  }
  // Basic training: extract the nested 'hyperparameters' dict (actual model params)
  const nested = hp.hyperparameters;
  if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }
  return {};
};

/** Resolve the terminal node configuration without merging fallback sources. */
export function getJobConfig(job: JobInfo): Config | null {
  const primary = hasTuningMetadata(job) ? job.config : job.hyperparameters;
  return (primary as Config) ||
    (job.graph?.nodes as GraphNode[] | undefined)?.find(n => n.node_id === job.node_id)?.params || null;
}

/** Keep first-appearance order when combining customized model keys. */
export function getParameterKeys(selectedJobs: JobInfo[]): string[] {
  return Array.from(new Set(selectedJobs.flatMap(job => Object.keys(getModelParams(job)))));
}
