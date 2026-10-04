import { CVPolicySettings, policyChange, temporalPolicy } from '../components/CVPolicySettings';
import { ValidationField } from '../../../../components/shared/ValidationField';
import { numericDraft, numericInputValue } from '../../../../core/utils/numericValidation';
import { BarChart3, ChevronRight } from 'lucide-react';
import { HelpTooltip } from '../components/HelpTooltip';
import { NestedFoldSettings } from '../components/NestedFoldSettings';
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
                    {config.cv_type === 'nested_cv'
                        ? 'Nested CV evaluates independent searches on untouched outer folds. Preprocessing is fitted within each training fold.'
                        : isAdvanced
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
                <label htmlFor={`${fieldId}-cv-folds`} className="block text-xs text-gray-500 mb-1">{config.cv_type === 'nested_cv' ? 'Outer folds' : 'Folds'}</label>
                <ValidationField field="cv_folds"><input
                    id={`${fieldId}-cv-folds`}
                    type="number"
                    value={numericInputValue(config.cv_folds, 5)}
                    onChange={(e) => onChange({ ...config, cv_folds: numericDraft(e.target.value) })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    min={2}
                /></ValidationField>
            </div>
            <div>
                <label htmlFor={`${fieldId}-cv-method`} className="block text-xs text-gray-500 mb-1">Method</label>
                <select
                    id={`${fieldId}-cv-method`}
                    value={config.cv_type ?? 'k_fold'}
                    onChange={(e) => onChange({ ...config, ...policyChange('cv_type', e.target.value, config) })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                >
                    <option value="k_fold">K-Fold</option>
                    <option value="stratified_k_fold">Stratified</option>
                    <option value="time_series_split">Time Series</option>
                    <option value="shuffle_split">Shuffle Split</option>
                    <option value="nested_cv">Nested CV</option>
            <option value="group_k_fold">Group K-Fold</option>
            <option value="stratified_group_k_fold">Stratified Group K-Fold</option>
                </select>
            </div>
        </div>
        {config.cv_type === 'nested_cv' && (
            <NestedFoldSettings fieldId={fieldId} outerFolds={config.cv_folds} innerFolds={config.cv_inner_folds}
                onChange={(value) => { onChange({ ...config, cv_inner_folds: value }); }} />
        )}
        <CVPolicySettings config={config} fieldId={fieldId} columns={availableColumns} update={patch => onChange({ ...config, ...patch })} />
        <div className="flex items-center gap-2">
            <input
                type="checkbox"
                id={`${fieldId}-cv_shuffle`}
                checked={config.cv_shuffle !== false}
          disabled={temporalPolicy(config)}
                onChange={(e) => onChange({ ...config, cv_shuffle: e.target.checked })}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor={`${fieldId}-cv_shuffle`} className="text-xs text-gray-600 dark:text-gray-400">Shuffle Data</label>
        </div>
        {config.cv_shuffle !== false && !temporalPolicy(config) && (
            <div>
                <div className="flex items-center gap-1.5 mb-1">
                    <label htmlFor={`${fieldId}-cv-seed`} className="block text-xs text-gray-500">Fold Split Seed</label>
                    <HelpTooltip text="Seed controlling how rows are dealt to folds — same seed = identical fold splits, so CV scores stay comparable across runs." />
                </div>
                <ValidationField field="cv_random_state"><input
                    id={`${fieldId}-cv-seed`}
                    type="number"
                    value={numericInputValue(config.cv_random_state, 42)}
                    onChange={(e) => onChange({ ...config, cv_random_state: numericDraft(e.target.value) })}
                    className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    min={0}
                /></ValidationField>
            </div>
        )}
    </div>
  );
}
