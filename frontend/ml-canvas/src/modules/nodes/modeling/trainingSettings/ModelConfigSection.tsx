import React from 'react';
import { ChevronDown, AlertCircle, AlertTriangle } from 'lucide-react';
import { ValidationField } from '../../../../components/shared/ValidationField';
import type { TrainingRunMode } from '../TrainingSettings';
import { TuningStrategySection } from './TuningStrategySection';
import { CrossValidationSection } from './CrossValidationSection';
import type { TrainingSettingsState } from './useTrainingSettings';
/** Two-option segmented control matching `EnsembleSettings`'s run-mode toggle. */
const RunModeToggle: React.FC<{ value: TrainingRunMode; onSelect: (v: TrainingRunMode) => void }> = ({ value, onSelect }) => (
  <div role="group" aria-label="Training Mode" className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
    {([['basic', 'Basic'], ['advanced', 'Advanced (Tuning)']] as const).map(([v, label]) => (
      <button
        key={v}
        type="button"
        aria-pressed={value === v}
        onClick={() => { onSelect(v); }}
        className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
          value === v
            ? 'bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-300 shadow-sm'
            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
        }`}
      >
        {label}
      </button>
    ))}
  </div>
);

type ModelConfigSectionProps = Pick<TrainingSettingsState,
  'config'
  | 'onChange'
  | 'isAdvanced'
  | 'fieldId'
  | 'availableModels'
  | 'isLoadingModels'
  | 'requiresScaling'
  | 'showScalingAlert'
  | 'setShowScalingAlert'
  | 'keepCustomizationOpen'
  | 'availableColumns'
  | 'isClassification'
  | 'setShowStrategyModal'
  | 'showCV'
  | 'setShowCV'>;

export function ModelConfigSection({
  config,
  onChange,
  isAdvanced,
  fieldId,
  availableModels,
  isLoadingModels,
  requiresScaling,
  showScalingAlert,
  setShowScalingAlert,
  keepCustomizationOpen,
  availableColumns,
  isClassification,
  setShowStrategyModal,
  showCV,
  setShowCV,
}: ModelConfigSectionProps) {
  return (
    <div className="space-y-5 animate-in fade-in duration-300">
        {/* Mode toggle */}
        <div className="space-y-1.5">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Training Mode</span>
            <RunModeToggle
                value={config.run_mode}
                onSelect={(v) => { onChange({ ...config, run_mode: v }); }}
            />
        </div>

        {/* Model & Target */}
        <div className="space-y-4">
            <div className="space-y-1.5">
                <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</span>
                <div className="grid gap-3">
                    <ValidationField field="model_type">
                        <label htmlFor={`${fieldId}-model`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Model Type</label>
                        <div className="relative">
                            <select
                                id={`${fieldId}-model`}
                                value={config.model_type}
                                onChange={(e) => {
                                    if (isAdvanced) {
                                        onChange({ ...config, model_type: e.target.value, search_space: {} });
                                        return;
                                    }
                                    // Check if customization is active before switching
                                    if (Object.keys(config.hyperparameters).length > 0) {
                                        keepCustomizationOpen.current = true;
                                    }
                                    onChange({ ...config, model_type: e.target.value, hyperparameters: {} });
                                }}
                                className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                                disabled={isLoadingModels}
                            >
                                {availableModels.map(model => (
                                    <option key={model.id} value={model.id}>{model.name}</option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
                        </div>

                        {requiresScaling && (
                           <div className="mt-2 text-xs border border-blue-200 dark:border-blue-800 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 overflow-hidden transition-all">
                               <button
                                   aria-expanded={showScalingAlert}
                                   onClick={() => setShowScalingAlert(!showScalingAlert)}
                                   className="w-full flex items-center justify-between p-2 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
                               >
                                   <div className="flex items-center gap-2 font-semibold">
                                        <AlertCircle className="w-3 h-3" />
                                        <span>Scale Your Data</span>
                                   </div>
                                    <ChevronDown className={`w-3 h-3 transition-transform ${showScalingAlert ? 'rotate-180' : ''}`} />
                               </button>
                               {showScalingAlert && (
                                   <div className="p-2 pt-0 opacity-90 animate-in slide-in-from-top-1 pl-7">
                                       This model performs best with scaled features. Consider adding a &quot;Feature Scaling&quot; node.
                                   </div>
                               )}
                           </div>
                        )}
                    </ValidationField>

                    <ValidationField field="target_column">
                        <label htmlFor={`${fieldId}-target`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</label>
                        {availableColumns.length === 0 && (
                            <div className="mb-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-xs text-yellow-700 dark:text-yellow-400 flex items-center gap-2">
                                <AlertTriangle className="w-3 h-3" />
                                <span>Connect a dataset node to see available columns</span>
                            </div>
                        )}
                        <div className="relative">
                            {availableColumns.length > 0 ? (
                                <select
                                    id={`${fieldId}-target`}
                                    value={config.target_column}
                                    onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                                    className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                >
                                    <option value="">Select target...</option>
                                    {availableColumns.map((col) => (
                                        <option key={col.name} value={col.name}>{col.name}</option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    id={`${fieldId}-target`}
                                    type="text"
                                    value={config.target_column}
                                    onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                                    className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                    placeholder="e.g., target"
                                />
                            )}
                            {availableColumns.length > 0 && <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />}
                        </div>
                    </ValidationField>
                </div>
            </div>

            {isAdvanced && (
                <TuningStrategySection config={config} onChange={onChange} fieldId={fieldId} isClassification={isClassification} setShowStrategyModal={setShowStrategyModal} />
            )}

            <div className="border-t border-gray-100 dark:border-gray-700" />

            {/* CV Settings */}
            <CrossValidationSection config={config} onChange={onChange} fieldId={fieldId} isAdvanced={isAdvanced} showCV={showCV} setShowCV={setShowCV} availableColumns={availableColumns} />
        </div>
    </div>
  );
}
