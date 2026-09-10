import type { ReactNode } from 'react';
import type { EvaluationData } from '../../types';
import { LoadingState, ErrorState } from '../../../../shared';

/** Render fetch and artifact availability states before the ready summary. */
export function SegmentationStatus({ evalError, evaluationData, isEvalLoading, retryJobId, fetchEvaluationData, hasSummary, children }: {
  evalError: string | null;
  evaluationData: EvaluationData | null;
  isEvalLoading: boolean;
  retryJobId: string | null;
  fetchEvaluationData: (jobId: string) => void | Promise<void>;
  hasSummary: boolean;
  children: ReactNode;
}) {
  return <>
    {evalError ? (
      <div className="h-64 flex items-center justify-center">
        <ErrorState
          error={evalError}
          onRetry={retryJobId ? () => fetchEvaluationData(retryJobId) : undefined}
        />
      </div>
    ) : !evaluationData ? (
      isEvalLoading ? (
        <div className="h-64 flex items-center justify-center">
          <LoadingState message="Loading segmentation data..." />
        </div>
      ) : (
        <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
          <p>Select a completed Segmentation run to view cluster details.</p>
        </div>
      )
    ) : evaluationData.problem_type !== 'clustering' ? (
      <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
        <p>The selected run is not a Segmentation (clustering) job.</p>
        <p className="text-xs mt-1 not-italic">See the availability list above for which selected runs do support clustering.</p>
      </div>
    ) : !hasSummary ? (
      <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
        <p>No cluster summary available for this run.</p>
        <p className="text-xs mt-1 not-italic">This run supports clustering but hasn&apos;t produced a summary yet — see the availability list above for details.</p>
      </div>
    ) : children}
  </>;
}

/** Read a split only from clustering payloads. */
export function getClusteringSplit(evaluationData: EvaluationData | null, currentSplitName: string | null) {
  return evaluationData && evaluationData.problem_type === 'clustering' && currentSplitName
    ? (evaluationData.splits[currentSplitName] ?? null)
    : null;

}
