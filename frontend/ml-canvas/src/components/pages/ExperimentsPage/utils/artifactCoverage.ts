/**
 * Pure helpers for resolving whether a selected run supports a given
 * explainability/segmentation artifact (feature importance, SHAP, or
 * clustering summary), and if so whether it's available, still pending, or
 * failed — so a missing tab or a zero-like bar is never the only signal a
 * user gets about artifact availability (UX finding EXP-003).
 */
import type { ExperimentsTask } from './jobMeta';

export type ArtifactKind = 'feature_importance' | 'shap' | 'segmentation';

/**
 * - `available`: the run's task supports this artifact and it was computed.
 * - `unsupported`: the run's task/model family never produces this artifact.
 * - `not_computed`: the task supports it, but it wasn't produced (older run,
 *   trainer skipped it, or the run hasn't finished yet).
 * - `failed`: the run errored before this artifact could be produced.
 */
export type ArtifactStatus = 'available' | 'unsupported' | 'not_computed' | 'failed';

export interface ArtifactCoverageInput {
  task: ExperimentsTask;
  /** The job's own lifecycle status string (e.g. "completed", "failed", "running"). */
  status: string;
  error?: string | null;
  /** Whether this artifact's data was actually found on the job's result. */
  hasArtifact: boolean;
}

export interface ArtifactCoverageResult {
  status: ArtifactStatus;
  reason: string;
}

const SUPPORTED_TASKS: Record<ArtifactKind, ExperimentsTask[]> = {
  feature_importance: ['classification', 'regression', 'text_classification', 'ensemble'],
  shap: ['classification', 'regression', 'text_classification', 'ensemble'],
  segmentation: ['segmentation'],
};

const ARTIFACT_LABEL: Record<ArtifactKind, string> = {
  feature_importance: 'Feature importance',
  shap: 'SHAP explanation',
  segmentation: 'Clustering summary',
};

const FAILED_STATUSES = new Set(['failed']);
const TERMINAL_SUCCESS_STATUSES = new Set(['completed', 'succeeded']);

/**
 * Resolves a single run's coverage status and a human-readable reason for a
 * given artifact kind, in the precedence order: failed run > run still in
 * progress > artifact present (available) > unsupported task > artifact
 * missing on a completed run.
 *
 * The task-support list is an *expectation* used only to explain an absent
 * artifact, never to deny a present one: presence is ground truth, and an
 * `'other'` task (model task unresolvable, e.g. registry not loaded yet)
 * must not be reported as "unsupported" (F-42).
 */
export function getArtifactCoverage(kind: ArtifactKind, input: ArtifactCoverageInput): ArtifactCoverageResult {
  const label = ARTIFACT_LABEL[kind];

  if (FAILED_STATUSES.has(input.status) || input.error) {
    return {
      status: 'failed',
      reason: input.error
        ? `Run failed before this artifact could be produced: ${input.error}`
        : 'Run failed before this artifact could be produced.',
    };
  }

  if (!TERMINAL_SUCCESS_STATUSES.has(input.status)) {
    return {
      status: 'not_computed',
      reason: 'Run has not finished yet — this artifact is not available until it completes.',
    };
  }

  if (input.hasArtifact) {
    return { status: 'available', reason: `${label} is available for this run.` };
  }

  if (!SUPPORTED_TASKS[kind].includes(input.task) && input.task !== 'other') {
    return {
      status: 'unsupported',
      reason: kind === 'segmentation'
        ? `Not a Segmentation (clustering) run — ${label.toLowerCase()} does not apply to this task.`
        : `${label} does not apply to this run's task/model family.`,
    };
  }

  return {
    status: 'not_computed',
    reason: `${label} was not computed for this run — not supported for this model type, or this run predates support for it.`,
  };
}
