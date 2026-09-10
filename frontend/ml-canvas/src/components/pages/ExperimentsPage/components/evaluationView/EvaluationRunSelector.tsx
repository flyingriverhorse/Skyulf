import type { EvaluationViewProps } from './types';
import type { EvaluationPresentation } from './useEvaluationPresentation';

/** Keep selection labels and callbacks in the supplied run order. */
export function EvaluationRunSelector({ evalJobId, fetchEvaluationData, eligibleRunLabels }:
  Pick<EvaluationViewProps, 'evalJobId' | 'fetchEvaluationData'> & Pick<EvaluationPresentation, 'eligibleRunLabels'>) {
  return <>
    {eligibleRunLabels.length > 1 && (
      <div
        className="flex gap-2 overflow-x-auto pb-2"
        role="tablist"
        aria-label="Select run for evaluation"
      >
        {eligibleRunLabels.map(({ jobId, label }) => {
          const isActive = evalJobId === jobId;
          return (
            <button
              key={jobId}
              type="button"
              role="tab"
              aria-selected={isActive}
              title={isActive ? `Active run: ${label}` : `Switch to run ${label}`}
              onClick={() => { void fetchEvaluationData(jobId); }}
              className={`px-3 py-1 text-xs font-mono rounded border whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${isActive
                ? 'bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300'
                : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700'
                }`}
            >
              {label}
            </button>
          );
        })}
      </div>
    )}
  </>;
}
