import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import { THRESHOLD_TUNING_METRICS } from '../../../../../core/api/thresholdTuning';
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import type { EvaluationViewProps } from './types';
import type { ThresholdMutationState } from './useThresholdMutation';

/** Unsupported saved metrics need a fresh preview before their cutoffs can be replaced. */
function requiresSupportedPreview(preview: EvaluationViewProps['tuningPreview']): boolean {
  return preview !== null && !THRESHOLD_TUNING_METRICS.includes(preview.metric);
}

/** Preview, save, enable and clear remain separate actions with shared busy feedback. */
export function ThresholdTuningControls({
  selectedTuningMetric,
  onSelectedTuningMetricChange,
  tuningPreview,
  hasSavedThresholds,
  useTunedThresholds,
  onPreviewThresholds,
  onSaveThresholds,
  onToggleThresholds,
  onClearThresholds,
  runThresholdMutation,
  isThresholdMutationPending,
  pendingThresholdMutationText,
  thresholdMutationError,
  thresholdMutationRetryLabel,
  handleRetryThresholdMutation,
}: EvaluationViewProps & ThresholdMutationState) {
  const needsSupportedPreview = requiresSupportedPreview(tuningPreview);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">Threshold Tuning</h4>
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Metric:</span>
        <select
          aria-label="Threshold tuning metric"
          className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
          value={selectedTuningMetric}
          onChange={(e) => { onSelectedTuningMetricChange(e.target.value); }}
        >
          <option value="accuracy">Accuracy</option>
          <option value="f1">F1</option>
          <option value="precision">Precision</option>
          <option value="recall">Recall</option>
          <option value="balanced_accuracy">Balanced Accuracy</option>
        </select>
        <InfoTooltip
          text="Which class-prediction metric the optimizer maximizes when you click Preview. Uses validation if available, otherwise test. ROC AUC measures probability ranking and does not change with the decision threshold."
          align="center"
        />
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={() => { void runThresholdMutation('preview', onPreviewThresholds); }}
          disabled={isThresholdMutationPending}
          className="px-3 py-1.5 rounded-lg text-sm font-medium action-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Preview
        </button>
        <InfoTooltip
          text="Runs the optimizer now and shows the result below — does not save or affect real predictions yet."
          align="center"
        />
      </div>
      {tuningPreview && (
        <>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Computed from {tuningPreview.split_used} split
            {tuningPreview.split_used === 'test' && (
              <em className="ml-1">(no validation split available — using test split)</em>
            )}
            {tuningPreview.source === 'training' && (
              <span
                className="ml-1.5 inline-flex items-center rounded-full bg-indigo-50 dark:bg-indigo-900/30 border border-indigo-200 dark:border-indigo-800 px-1.5 py-0.5 text-[10px] font-medium text-indigo-600 dark:text-indigo-300"
                title="These thresholds were selected automatically during model training (Tune decision threshold) and saved here — you can preview new ones, save over them, or clear them."
              >
                seeded at training
              </span>
            )}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => { void runThresholdMutation('save', onSaveThresholds); }}
              disabled={needsSupportedPreview || isThresholdMutationPending}
              className="px-3 py-1.5 rounded-lg text-sm font-medium action-secondary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
            <InfoTooltip
              text="Saves these thresholds to this model version and enables them for predictions."
              align="center"
            />
          </div>
        </>
      )}
      <div className="flex items-center gap-1">
        <label className="flex items-center gap-1.5 cursor-pointer text-sm">
          <input
            type="checkbox"
            checked={useTunedThresholds}
            onChange={(e) => {
              const enabled = e.target.checked;
              void runThresholdMutation(enabled ? 'enable' : 'disable', () => onToggleThresholds(enabled));
            }}
            disabled={!hasSavedThresholds || isThresholdMutationPending}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
          />
          <span className="text-gray-700 dark:text-gray-300">Use tuned thresholds at prediction time</span>
        </label>
        <InfoTooltip
          text="When ON, every real /predict call for this model uses these saved thresholds instead of the default 0.5/argmax rule."
          align="center"
        />
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={() => { void runThresholdMutation('clear', onClearThresholds); }}
          disabled={isThresholdMutationPending}
          className="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Clear
        </button>
        <InfoTooltip
          text="Deletes saved thresholds entirely and reverts predictions to the default rule."
          align="center"
        />
      </div>
      {!hasSavedThresholds && (
        <p className="w-full text-xs text-gray-500 dark:text-gray-400">
          Preview thresholds, then Save to enable them for predictions.
        </p>
      )}
      {needsSupportedPreview && (
        <p className="w-full text-xs text-gray-500 dark:text-gray-400">
          This saved set uses a metric unavailable for new tuning. Choose a metric and click Preview before saving a replacement.
        </p>
      )}
      {pendingThresholdMutationText && (
        <span className="inline-flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400" role="status" aria-atomic="true">
          <Loader2 className="w-3 h-3 animate-spin" aria-hidden="true" />
          {pendingThresholdMutationText}
        </span>
      )}
      {thresholdMutationError && (
        <div className="inline-flex items-center gap-2 rounded-md border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 px-2 py-1 text-xs text-red-700 dark:text-red-300" role="alert" aria-atomic="true">
          <AlertTriangle className="w-3 h-3 shrink-0" aria-hidden="true" />
          <span>{thresholdMutationError.message}</span>
          <button
            type="button"
            onClick={handleRetryThresholdMutation}
            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-medium text-red-700 dark:text-red-200 hover:bg-red-100 dark:hover:bg-red-900/40 disabled:opacity-50"
          >
            <RefreshCw className="w-3 h-3" aria-hidden="true" />
            {thresholdMutationRetryLabel}
          </button>
        </div>
      )}
    </div>
  );
}
