/**
 * Lightweight helpers for working with job records on the Experiments page.
 * Pure functions only — no React, no API calls.
 */
import type { TaskType } from '../../../../core/types/taskType';

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

/**
 * Task-type filter values shown as Job History / Experiments tabs
 * (plan §0.5/§0.6: group jobs by task, not by basic_training/advanced_tuning
 * engine). "other" covers any model_type not resolvable via registry tags
 * (e.g. registry not loaded yet, or a model added without task tags).
 */
export type ExperimentsTask = TaskType | 'other';

/**
 * Resolves a job's task type from its `model_type`, using the same
 * `RegistryItem.tags` data `TrainingSettings.tsx` uses to filter each task
 * node's model dropdown (clustering \u2192 segmentation, classification \u2192
 * classification, regression \u2192 regression, text/nlp \u2192 text_classification).
 *
 * `logistic_regression` is dual-tagged (`classification` + `text`/`nlp`,
 * since it's usable directly on vectorized text features too) \u2014 with no
 * other signal to disambiguate, this defaults it to `classification` (the
 * more common case). Naive Bayes (`multinomial_nb`/`bernoulli_nb`) and
 * `sgd_classifier` are tagged `text`/`nlp` only, so they always resolve to
 * `text_classification`.
 */
export function getTaskForModelType(
  modelType: string | undefined,
  registryItems: { id: string; tags?: string[] }[],
): ExperimentsTask {
  if (!modelType) return 'other';
  const tags = registryItems.find(r => r.id === modelType)?.tags ?? [];
  if (tags.includes('clustering')) return 'segmentation';
  if (modelType === 'logistic_regression') return 'classification';
  if (tags.includes('text') || tags.includes('nlp')) return 'text_classification';
  if (tags.includes('classification')) return 'classification';
  if (tags.includes('regression')) return 'regression';
  return 'other';
}

/**
 * Whether a job actually carries tuning metadata (search strategy/best
 * params from an advanced-tuning run), rather than checking `job_type ===
 * 'advanced_tuning'` directly (plan §0.5: task type and tuning metadata are
 * decoupled concepts from the job's `job_type` string). `search_strategy`
 * is the reliable signal here \u2014 it's a column that only exists on the
 * backend's `AdvancedTuningJob` model (`nullable=False` there), so it's
 * always populated for tuned runs and always absent/undefined for basic
 * training runs, independent of the `job_type` string.
 */
export function hasTuningMetadata(job: { job_type?: string; search_strategy?: string }): boolean {
  return job.search_strategy != null || job.job_type === 'advanced_tuning';
}
