import { Loader2, Play } from 'lucide-react';
import { RunFeedback } from '../../../../components/shared/RunFeedback';
import { TrainingActionFooter } from '../../../../components/shared/TrainingActionFooter';
import type { RegistryItem } from '../../../../core/api/registry';
import type { SubmittedRun } from '../../../../core/types/runFeedback';
import type { SegmentationConfig } from '../SegmentationSettings';

/** Describe action availability and retain the node's pending or completed receipt. */
export function SegmentationActionFooter({
  runHelpId, datasetId, config, selectedModelItem, isSubmitting, handleTrain, submissionMessage, runFeedback,
}: {
  runHelpId: string;
  datasetId: string | undefined;
  config: SegmentationConfig;
  selectedModelItem: RegistryItem | undefined;
  isSubmitting: boolean;
  handleTrain: () => Promise<void>;
  submissionMessage: string;
  runFeedback: SubmittedRun | null | undefined;
}) {
  return (
    <TrainingActionFooter details={
      <p id={runHelpId} className="text-xs text-center text-muted-foreground">
        {!datasetId ? 'Connect a dataset node upstream and select a dataset to enable this action.'
          : !config.model_type ? 'Choose a clustering algorithm to enable this action.'
          : `Trains ${selectedModelItem?.name || config.model_type.replace(/_/g, ' ')} in the background without a target column.`}
      </p>
    }>
      <button
        type="button"
        onClick={() => { void handleTrain(); }}
        disabled={!datasetId || !config.model_type || isSubmitting}
        aria-describedby={runHelpId}
        className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 action-primary rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-lg disabled:hover:translate-y-0 focus-ring"
      >
        {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
        <span className="text-sm font-semibold">{isSubmitting ? 'Submitting job...' : 'Train segmentation'}</span>
      </button>
      {submissionMessage && <p role="status" aria-atomic="true" className="text-xs text-center text-muted-foreground break-words">{submissionMessage}</p>}
      {runFeedback && <RunFeedback run={runFeedback} task="segmentation" />}
    </TrainingActionFooter>
  );
}
