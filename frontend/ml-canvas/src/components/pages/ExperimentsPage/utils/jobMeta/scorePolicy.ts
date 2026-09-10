import type { DisplayScore } from '../jobMeta';

/** Macro/micro rates have no faithful threshold-scan equivalent. */
export function isThresholdRate(metric: string, rate: string): boolean {
  return metric.startsWith(rate) && !metric.includes('macro') && !metric.includes('micro');
}

/** Tuning reports its chosen metric and cross-validation score directly. */
export function getTuningDisplayScore(job: { result?: Record<string, unknown> | null; config?: unknown }): DisplayScore | null {
  const best = (job.result as { best_score?: unknown } | undefined)?.best_score;
  if(typeof best === 'number' && !Number.isNaN(best)) {
    const scoring = getJobScoringMetric(job) || 'score';
    return { metric: scoring, value: best, split: 'cv' };
  }
  return null;
}

/**
 * Extract the resolved scoring metric from a job's result (top-level or
 * nested in metrics), falling back to the tuning config's requested metric
 * when a job hasn't finished (or errored) before `scoring_metric` was
 * recorded onto the result.
 */
export function getJobScoringMetric(job: { result?: Record<string, unknown> | null; config?: unknown }): string | undefined {
  const r = job.result;
  if(r?.scoring_metric) return r.scoring_metric as string;
  const m = r?.metrics as Record<string, unknown> | undefined;
  if(m?.scoring_metric) return m.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  if(typeof tuning?.metric === 'string') return tuning.metric;
  return undefined;
}
