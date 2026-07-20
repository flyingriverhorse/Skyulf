/**
 * Lightweight helpers for working with job records on the Experiments page.
 * Pure functions only — no React, no API calls.
 */
import type { TaskType } from '../../../../core/types/taskType';
import type { ThresholdMetric } from './classificationCharts';
export type { ThresholdMetric };

/**
 * Extract the resolved scoring metric from a job's result (top-level or
 * nested in metrics), falling back to the tuning config's requested metric
 * when a job hasn't finished (or errored) before `scoring_metric` was
 * recorded onto the result.
 */
export function getJobScoringMetric(job: { result?: Record<string, unknown> | null; config?: unknown }): string | undefined {
  const r = job.result;
  if (r?.scoring_metric) return r.scoring_metric as string;
  const m = r?.metrics as Record<string, unknown> | undefined;
  if (m?.scoring_metric) return m.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  if (typeof tuning?.metric === 'string') return tuning.metric;
  return undefined;
}

/**
 * Maps a job's own scoring metric (as reported by `getJobScoringMetric`,
 * e.g. "f1_weighted", "roc_auc", "precision_weighted") to the closest
 * dropdown option, so the Model Evaluation metric selector defaults to
 * whatever the job was actually scored/tuned on instead of always F1.
 * Unmappable/threshold-independent metrics (e.g. "roc_auc",
 * "balanced_accuracy") fall back to "f1_weighted" — a safe default that
 * works for both binary and multiclass jobs.
 */
export function mapJobMetricToDropdown(scoringMetric: string | undefined): ThresholdMetric {
  if (!scoringMetric) return 'f1_weighted';
  const m = scoringMetric.toLowerCase();
  if (m === 'accuracy') return 'accuracy';
  if (m === 'f1_weighted') return 'f1_weighted';
  if (m === 'f1' || m === 'f1_macro') return 'f1';
  if (m.startsWith('precision')) return 'precision';
  if (m.startsWith('recall')) return 'recall';
  return 'f1_weighted';
}

/** Per-task priority list of metric base names, best-first, used by `getDisplayScore`. */
const SCORE_METRIC_PRIORITY: Record<'classification' | 'text_classification' | 'regression' | 'segmentation', string[]> = {
  classification: ['f1_weighted', 'accuracy', 'f1_macro', 'f1', 'roc_auc'],
  text_classification: ['f1_weighted', 'accuracy', 'f1_macro', 'f1', 'roc_auc'],
  regression: ['r2', 'rmse', 'mae'],
  // n_clusters is intentionally excluded — it's a cluster count, not a
  // quality score, so it should never be chosen as the headline metric.
  segmentation: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
};

// Preferred split order when reading a `basic_training` job's flat
// `{split}_{metric}` keys (e.g. `test_accuracy`). Mirrors the backend's own
// headline-metric search order in `_execution/summary.py`: held-out
// performance (test) is more meaningful than train-set performance, which
// can look artificially high due to overfitting.
const SCORE_SPLIT_PRIORITY = ['test', 'val', 'train'] as const;

export interface DisplayScore {
  /** Metric base name, e.g. "f1_weighted", "r2", "silhouette_score". */
  metric: string;
  value: number;
  /** Which split the value came from ("cv" for tuning jobs' cross-validated best_score). */
  split: 'test' | 'val' | 'train' | 'cv';
}

/**
 * Resolves the single most meaningful score to headline for a job, picking
 * a task-appropriate metric (never a raw count like `n_clusters` for
 * segmentation) and preferring held-out splits (test > val > train) over
 * the train split alone.
 *
 * - `advanced_tuning` jobs: use the cross-validated `best_score` + the
 *   scoring metric that was actually optimized (unambiguous, already
 *   task-appropriate since the user picked it).
 * - `basic_training` jobs: scan `result.metrics`'s flat `{split}_{metric}`
 *   keys using the task's metric priority list and the split priority
 *   above, so e.g. a classification job shows `test_f1_weighted` before
 *   falling back to `train_accuracy`.
 */
export function getDisplayScore(
  job: { job_type: string; result?: Record<string, unknown> | null; config?: unknown },
  task: ExperimentsTask,
): DisplayScore | null {
  if (job.job_type === 'advanced_tuning') {
    const best = (job.result as { best_score?: unknown } | undefined)?.best_score;
    if (typeof best === 'number' && !Number.isNaN(best)) {
      const scoring = getJobScoringMetric(job) || 'score';
      return { metric: scoring, value: best, split: 'cv' };
    }
    return null;
  }

  const metrics = (job.result as { metrics?: Record<string, unknown> } | undefined)?.metrics;
  if (!metrics) return null;

  const priority = task === 'other' ? [] : SCORE_METRIC_PRIORITY[task];
  for (const base of priority) {
    for (const split of SCORE_SPLIT_PRIORITY) {
      const v = metrics[`${split}_${base}`];
      if (typeof v === 'number' && !Number.isNaN(v)) {
        return { metric: base, value: v, split };
      }
    }
  }
  return null;
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
