import type { ToolbarState } from './_hooks/useToolbarState';
import { Play, Loader2, Rocket } from 'lucide-react';

export function ToolbarExperimentControl(
  { layout, run, menus, readOnly }: Pick<ToolbarState, 'layout' | 'run' | 'menus' | 'readOnly'>,
) {
  const { isNarrow } = layout;
  const { isRunning, isRunningAll, hasMultipleBranches } = run;
  const { setReviewingExperiments } = menus;
  return (<>
    {!readOnly && !isNarrow && hasMultipleBranches && (
      <button
        onClick={() => setReviewingExperiments(true)}
        disabled={isRunningAll || isRunning}
        title="Run all branches as separate experiments"
        aria-label="Run all parallel branches as separate experiments"
        data-testid="toolbar-run-all"
        className="flex items-center gap-2 px-3 py-2 action-primary rounded-md shadow-sm transition-colors disabled:opacity-50"
      >
        {isRunningAll ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Rocket className="w-4 h-4" />
        )}
        <span className="text-sm font-medium hidden 2xl:inline">
          {isRunningAll ? 'Queuing...' : 'Run All Experiments'}
        </span>
      </button>
    )}
  </>);
}

export function ToolbarPreviewControl(
  { run, readOnly }: Pick<ToolbarState, 'run' | 'readOnly'>,
) {
  const { isRunning, handleRun } = run;
  return (<>
    {!readOnly && (
      <button
        onClick={() => { void handleRun(); }}
        disabled={isRunning}
        title="Preview data (Ctrl+Enter). Click to review any blocking issues."
        data-testid="toolbar-run-preview"
        className="flex shrink-0 items-center gap-2 px-3 py-2 action-primary rounded-md shadow-sm transition-all disabled:opacity-50 focus-ring"
      >
        {isRunning ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Play className="w-4 h-4" />
        )}
        <span className="text-sm font-medium whitespace-nowrap">
          {isRunning ? 'Previewing data...' : 'Preview data'}
        </span>
      </button>
    )}
  </>);
}
