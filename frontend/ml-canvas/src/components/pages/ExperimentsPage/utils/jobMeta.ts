/**
 * Lightweight helpers for working with job records on the Experiments page.
 * Pure functions only — no React, no API calls.
 */

/** Extract the resolved scoring metric from a job's result (top-level or nested in metrics). */
export function getJobScoringMetric(job: { result?: Record<string, unknown> | null }): string | undefined {
  const r = job.result;
  if (r?.scoring_metric) return r.scoring_metric as string;
  const m = r?.metrics as Record<string, unknown> | undefined;
  if (m?.scoring_metric) return m.scoring_metric as string;
  return undefined;
}

/**
 * Short 8-char run ID derived from a job's pipeline_id.
 * Strips the "preview_" prefix and any "__branch_N" suffix so all
 * experiments from the same batch share the same display ID.
 */
export function shortRunId(job: { pipeline_id: string; parent_pipeline_id?: string | null }): string {
  const raw = job.parent_pipeline_id || job.pipeline_id;
  const clean = raw.replace(/^preview_/, '').replace(/__branch_.*$/, '');
  return clean.slice(0, 8);
}
