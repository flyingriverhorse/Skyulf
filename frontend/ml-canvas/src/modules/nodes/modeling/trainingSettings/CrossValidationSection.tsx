import { BarChart3, ChevronRight, AlertTriangle } from 'lucide-react';
import { HelpTooltip } from '../components/HelpTooltip';
import type { TrainingSettingsState } from './useTrainingSettings';

type CrossValidationSectionProps = Pick<TrainingSettingsState,
  'config'
  | 'onChange'
  | 'fieldId'
  | 'isAdvanced'
  | 'showCV'
  | 'setShowCV'
  | 'availableColumns'>;

export function CrossValidationSection({
  config,
  onChange,
  fieldId,
  isAdvanced,
  showCV,
  setShowCV,
  availableColumns,
}: CrossValidationSectionProps) {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <button
            aria-expanded={showCV}
            onClick={() => { setShowCV(!showCV); }}
            className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
            <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200">Cross Validation</span>
            </div>
            <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${showCV ? 'rotate-90' : ''}`} />
        </button>

        {showCV && (
            <div className="p-3 space-y-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 animate-in slide-in-from-top-2 duration-200">
                <div className="flex items-center gap-2 mb-2">
                    <input
                        type="checkbox"
                        id={`${fieldId}-cv_enabled`}
                        checked={config.cv_enabled !== false}
                        onChange={(e) => onChange({ ...config, cv_enabled: e.target.checked })}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <label htmlFor={`${fieldId}-cv_enabled`} className="text-sm text-gray-700 dark:text-gray-300">Enable Cross-Validation</label>
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 pl-6">
                    {isAdvanced
                        ? 'Candidates are already scored by CV during the search. This re-evaluates the winning model after tuning (full metric panel + fold-to-fold variance); it never changes the selected hyperparameters.'
                        : 'Runs a k-fold evaluation of the trained model after training. Evaluation only, it measures generalization and never changes the model or its hyperparameters.'}
                </p>

                {config.cv_enabled !== false && (
                    <CrossValidationOptions config={config} onChange={onChange} fieldId={fieldId} availableColumns={availableColumns} />
                )}
            </div>
        )}
    </div>
  );
}


type CrossValidationOptionsProps = Pick<TrainingSettingsState, 'config' | 'onChange' | 'fieldId' | 'availableColumns'>;

function CrossValidationOptions({
  config,
  onChange,
  fieldId,
  availableColumns,
}: CrossValidationOptionsProps) {
  return (
    <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
        <div className="grid grid-cols-2 gap-3">
            <div>
                <label htmlFor={`${fieldId}-cv-folds`} className="block text-xs text-gray-500 mb-1">Folds</label>
                <input
                    id={`${fieldId}-cv-folds`}
                    type="number"
                    value={config.cv_folds ?? 5}
                    onChange={(e) => onChange({ ...config, cv_folds: Number(e.target.value) })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    min={2}
                />
            </div>
            <div>
                <label htmlFor={`${fieldId}-cv-method`} className="block text-xs text-gray-500 mb-1">Method</label>
                <select
                    id={`${fieldId}-cv-method`}
                    value={config.cv_type ?? 'k_fold'}
                    onChange={(e) => onChange({ ...config, cv_type: e.target.value })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                >
                    <option value="k_fold">K-Fold</option>
                    <option value="stratified_k_fold">Stratified</option>
                    <option value="time_series_split">Time Series</option>
                    <option value="shuffle_split">Shuffle Split</option>
                    <option value="nested_cv">Nested CV</option>
                </select>
            </div>
        </div>
        {config.cv_type === 'time_series_split' && (
            <div className="space-y-2">
                <div className="flex items-start gap-1.5 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs text-amber-700 dark:text-amber-400">
                    <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                    <span>Data must be sorted by time. Select a date column below or ensure your data is pre-sorted.</span>
                </div>
                <div>
                    <label htmlFor={`${fieldId}-cv-time-column`} className="block text-xs text-gray-500 mb-1">Time Column (optional)</label>
                    <select
                        id={`${fieldId}-cv-time-column`}
                        value={config.cv_time_column ?? ''}
                        onChange={(e) => onChange({ ...config, cv_time_column: e.target.value })}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    >
                        <option value="">Auto-detect</option>
                        {availableColumns
                            .filter((col) => {
                                const dt = String(col.dtype).toLowerCase();
                                return dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp');
                            })
                            .map((col) => (
                                <option key={col.name} value={col.name}>{col.name}</option>
                            ))
                        }
                        {availableColumns
                            .filter((col) => {
                                const dt = String(col.dtype).toLowerCase();
                                return !(dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp'));
                            })
                            .map((col) => (
                                <option key={col.name} value={col.name}>{col.name}</option>
                            ))
                        }
                    </select>
                </div>
            </div>
        )}
        <div className="flex items-center gap-2">
            <input
                type="checkbox"
                id={`${fieldId}-cv_shuffle`}
                checked={config.cv_shuffle !== false}
                onChange={(e) => onChange({ ...config, cv_shuffle: e.target.checked })}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor={`${fieldId}-cv_shuffle`} className="text-xs text-gray-600 dark:text-gray-400">Shuffle Data</label>
        </div>
        {config.cv_shuffle !== false && config.cv_type !== 'time_series_split' && (
            <div>
                <div className="flex items-center gap-1.5 mb-1">
                    <label htmlFor={`${fieldId}-cv-seed`} className="block text-xs text-gray-500">Fold Split Seed</label>
                    <HelpTooltip text="Seed controlling how rows are dealt to folds — same seed = identical fold splits, so CV scores stay comparable across runs." />
                </div>
                <input
                    id={`${fieldId}-cv-seed`}
                    type="number"
                    value={config.cv_random_state ?? 42}
                    onChange={(e) => onChange({ ...config, cv_random_state: Number(e.target.value) })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    min={0}
                />
            </div>
        )}
    </div>
  );
}
