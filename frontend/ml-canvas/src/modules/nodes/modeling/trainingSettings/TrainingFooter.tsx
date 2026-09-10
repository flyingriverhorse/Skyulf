import { Play, Loader2, Activity } from 'lucide-react';
import { TrainingActionFooter } from '../../../../components/shared/TrainingActionFooter';
import { RunFeedback } from '../../../../components/shared/RunFeedback';
import type { TrainingSettingsState } from './useTrainingSettings';

type TrainingFooterProps = Pick<TrainingSettingsState,
  'config'
  | 'isAdvanced'
  | 'selectedModelItem'
  | 'fieldId'
  | 'datasetId'
  | 'isSubmitting'
  | 'setShowParamsModal'
  | 'handleSubmit'
  | 'submissionMessage'
  | 'runFeedback'
  | 'historyTask'>;

export function TrainingFooter({
  config,
  isAdvanced,
  selectedModelItem,
  fieldId,
  datasetId,
  isSubmitting,
  setShowParamsModal,
  handleSubmit,
  submissionMessage,
  runFeedback,
  historyTask,
}: TrainingFooterProps) {
  return (
    <TrainingActionFooter details={<>
      <p className="text-xs text-center text-gray-600 dark:text-gray-400 break-words">
        Selected model: {selectedModelItem?.name || config.model_type.replace(/_/g, ' ')}
      </p>
      <p id={`${fieldId}-run-help`} className="text-xs text-center text-gray-600 dark:text-gray-400">
        {trainingRunHelp(config, datasetId, isAdvanced)}
      </p>

      {isAdvanced && (
          <button
              onClick={() => { setShowParamsModal(true); }}
              className="text-xs text-gray-500 hover:text-purple-600 dark:text-gray-400 dark:hover:text-purple-400 flex items-center gap-1.5 transition-colors px-3 py-1 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800"
          >
              <Activity className="w-3.5 h-3.5" />
              View Best Parameters History
          </button>
      )}
    </>}>
      <button
        type="button"
        onClick={() => { void handleSubmit(); }}
        disabled={isRunDisabled(config, datasetId, isSubmitting)}
        aria-describedby={`${fieldId}-run-help`}
        className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 action-primary rounded-lg shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus-ring"
      >
        {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
        <span className="text-sm font-semibold">{isSubmitting ? 'Submitting job...' : isAdvanced ? 'Tune model' : 'Train model'}</span>
      </button>
      {submissionMessage && <p role="status" aria-atomic="true" className="text-xs text-center text-muted-foreground break-words">{submissionMessage}</p>}
      {runFeedback && <RunFeedback run={runFeedback} task={historyTask} />}
    </TrainingActionFooter>
  );
}


function trainingRunHelp(config: TrainingSettingsState['config'], datasetId: string | undefined, isAdvanced: boolean): string {
    if (!datasetId) return 'Connect a dataset node upstream and select a dataset to enable this action.';
    if (!config.target_column?.trim()) return 'Choose a target column to enable this action.';
    if (!config.model_type) return 'Choose a model to enable this action.';
    return isAdvanced
        ? 'Searches hyperparameters and trains the selected model in the background.'
        : 'Trains the selected model with fixed parameters in the background.';
}

function isRunDisabled(config: TrainingSettingsState['config'], datasetId: string | undefined, isSubmitting: boolean): boolean {
    return !datasetId || isSubmitting || !config.target_column?.trim() || !config.model_type;
}
