/**
 * Selection bookkeeping for the Experiments page.
 *
 * Selections deliberately survive filter changes, which means a selected run
 * can be absent from the sidebar while still driving charts, tables, and the
 * evaluation tabs. These pure helpers make that split explicit so the page can
 * disclose it, and so a tab never targets a run it cannot render.
 */

import type { ExperimentsTask } from './jobMeta';

/** A selected run paired with its task and whether the active filter shows it. */
export interface SelectableRun {
  jobId: string;
  task: ExperimentsTask;
  /** Whether the run currently appears in the filtered sidebar list. */
  visible: boolean;
}

/** The two evaluation-style views that render a single run at a time. */
export type EvaluationView = 'evaluation' | 'segmentation';

/** Selected run ids split by whether the active filter shows them. */
export interface SelectionPartition {
  visible: string[];
  hidden: string[];
}

/** Split the current selection into filter-visible and filter-hidden runs. */
export function partitionSelection(runs: readonly SelectableRun[]): SelectionPartition {
  const visible: string[] = [];
  const hidden: string[] = [];
  for (const run of runs) {
    (run.visible ? visible : hidden).push(run.jobId);
  }
  return { visible, hidden };
}

/** Clustering runs have no supervised metrics; everything else has no cluster plots. */
function suitsView(view: EvaluationView, task: ExperimentsTask): boolean {
  return view === 'segmentation' ? task === 'segmentation' : task !== 'segmentation';
}

/**
 * The selected runs an evaluation-style tab can actually render, in selection
 * order. Filter-hidden runs are kept so the tab's run picker matches the set
 * driving the comparison.
 */
export function selectRunsForView(
  view: EvaluationView,
  runs: readonly SelectableRun[],
): string[] {
  return runs.filter((run) => suitsView(view, run.task)).map((run) => run.jobId);
}

/**
 * Decide which selected run an evaluation-style tab should display.
 *
 * Preference order: keep the current target if it is still selected and
 * renderable, otherwise the first compatible run the user can actually see,
 * otherwise the first compatible run at all.
 *
 * @returns The run to display, or `null` when no selected run suits the view.
 */
export function resolveEvaluationTarget(
  view: EvaluationView,
  runs: readonly SelectableRun[],
  currentTarget: string | null,
): string | null {
  const compatible = runs.filter((run) => suitsView(view, run.task));
  if (compatible.length === 0) return null;
  if (currentTarget !== null && compatible.some((run) => run.jobId === currentTarget)) {
    return currentTarget;
  }
  const firstVisible = compatible.find((run) => run.visible);
  return (firstVisible ?? compatible[0]!).jobId;
}
