import { LoadingState, ErrorState } from '../../../../shared';
import type { EvaluationViewProps, ChartEvaluationProps } from './types';
import type { EvaluationPresentation } from './useEvaluationPresentation';
import type { ThresholdMutationState } from './useThresholdMutation';
import { EvaluationControls } from './EvaluationControls';
import { ThresholdTuningControls } from './ThresholdTuningControls';
import { EvaluationCharts } from './EvaluationCharts';

type BodyProps = EvaluationViewProps & EvaluationPresentation & ThresholdMutationState;

/** Existing data stays mounted during background loading; errors take precedence. */
export function EvaluationBody(props: BodyProps) {
  const { evalError, evaluationData, isEvalLoading, fetchEvaluationData, evalJobId, eligibleJobIds } = props;
  const retryJobId = evalJobId ?? eligibleJobIds[0] ?? null;
  if (evalError) return (
    <div className="h-64 flex items-center justify-center">
      <ErrorState
        error={evalError}
        onRetry={retryJobId ? () => fetchEvaluationData(retryJobId) : undefined}
      />
    </div>
  );
  if (!evaluationData) return (
    isEvalLoading ? (
      <div className="h-64 flex items-center justify-center">
        <LoadingState message="Loading evaluation data..." />
      </div>
    ) : (
      <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
        <p>Select a completed job to view evaluation details.</p>
        <p className="text-xs mt-2 opacity-70">(Note: Only jobs run after this update have evaluation artifacts)</p>
      </div>
    )
  );
  if (evaluationData.problem_type === 'clustering') return (
    // Clustering jobs have no y_true/y_pred (only predicted cluster
    // labels), so the classification/regression charts below don't
    // apply — point the user at the dedicated Segmentation tab instead.
    <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
      <p>This is a Segmentation (clustering) run.</p>
      <p className="text-xs mt-2 opacity-70">See the &quot;Segmentation&quot; tab for cluster sizes, centroids, and quality metrics.</p>
    </div>
  );
  return <EvaluationContent {...props} evaluationData={evaluationData} />;
}

/** Present the controls and charts after the data kind has been checked. */
function EvaluationContent(props: BodyProps & ChartEvaluationProps) {
  const { evaluationData, isEvalLoading, activeTab } = props;
  return <div className={`space-y-6 transition-opacity ${isEvalLoading ? 'opacity-60' : ''}`}>
    {isEvalLoading && <div className="text-xs text-gray-500 dark:text-gray-400 italic">Loading evaluation data…</div>}
    <EvaluationControls {...props} />
    {/* Tab-level one-line description of what's driving the charts below */}
    {evaluationData.problem_type === 'classification' && (
      <p className="text-xs text-gray-500 dark:text-gray-400 italic px-1">
        {activeTab === 'slider'
          ? 'Manually explore how a single threshold changes predictions — nothing here is saved or used for real predictions.'
          : "Let the optimizer find the best per-class threshold(s) for a metric you choose, preview its effect, then save it to actually change how this model predicts."}
      </p>
    )}
    {activeTab === 'tuning' && evaluationData.problem_type === 'classification' && <ThresholdTuningControls {...props} />}
    <EvaluationCharts {...props} />
  </div>;
}
