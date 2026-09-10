import { Download, Loader2, Settings2 } from 'lucide-react';
import { HelpTooltip } from '../components/HelpTooltip';
import { HyperparameterInput } from '../components/HyperparameterInput';
import { SearchSpaceInput } from '../components/SearchSpaceInput';
import { isBasicParamVisible, isSearchSpaceParamVisible } from './modelOptions';
import type { TrainingSettingsState } from './useTrainingSettings';

type HyperparametersSectionProps = Pick<TrainingSettingsState,
  'config'
  | 'onChange'
  | 'fieldId'
  | 'hyperparameters'
  | 'isLoadingHyperparamDefs'
  | 'useCustomParams'
  | 'toggleCustomParams'
  | 'setShowParamsModal'>;

export function HyperparametersSection({
  config,
  onChange,
  fieldId,
  hyperparameters,
  isLoadingHyperparamDefs,
  useCustomParams,
  toggleCustomParams,
  setShowParamsModal,
}: HyperparametersSectionProps) {
  return (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="flex items-center justify-between">
           <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
             <Settings2 className="w-4 h-4 text-blue-500" />
             Hyperparameters
           </h4>
           <div className="flex items-center gap-2">
               <label className={`text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 cursor-pointer select-none ${isLoadingHyperparamDefs ? 'opacity-50 cursor-not-allowed' : ''}`}>
                   <input
                        type="checkbox"
                        checked={useCustomParams}
                        onChange={(e) => { toggleCustomParams(e.target.checked); }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        disabled={isLoadingHyperparamDefs}
                   />
                   Customize
               </label>
           </div>
        </div>

        {!useCustomParams ? (
            <div className="text-center py-8 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-dashed border-gray-200 dark:border-gray-700">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                    Using default hyperparameters.
                </p>
                <button
                    onClick={() => { setShowParamsModal(true); }}
                    className="mt-3 text-xs flex items-center gap-1.5 px-3 py-1.5 mx-auto bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-md border border-purple-200 dark:border-purple-800 hover:bg-purple-100 dark:hover:bg-purple-900/40 transition-colors shadow-sm"
                >
                    <Download className="w-3 h-3" />
                    Load Best Params
                </button>
            </div>
        ) : (
            <div className="space-y-3">
              <div className="flex justify-end mb-2">
                   <button
                      onClick={() => { setShowParamsModal(true); }}
                      className="text-xs flex items-center gap-1.5 px-2 py-1 text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 rounded transition-colors"
                    >
                      <Download className="w-3 h-3" />
                      Load Best
                   </button>
              </div>

              {isLoadingHyperparamDefs ? (
                  <div className="flex justify-center py-4">
                      <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                  </div>
              ) : (
                  hyperparameters.filter(param => isBasicParamVisible(param, config.hyperparameters)).map((param) => (
                    <div key={param.name} className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-2">
                          <label htmlFor={`${fieldId}-param-${param.name}`} className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                            {param.label}
                          </label>
                          {param.description && <HelpTooltip text={param.description} />}
                      </div>
                      {param.type === 'select' ? (
                        <select
                          id={`${fieldId}-param-${param.name}`}
                          value={(config.hyperparameters[param.name] ?? param.default) as string | number | readonly string[] | undefined}
                          onChange={(e) => onChange({
                            ...config,
                            hyperparameters: { ...config.hyperparameters, [param.name]: e.target.value }
                          })}
                          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        >
                          {param.options?.map((opt: { label: string; value: unknown }) => (
                            <option key={String(opt.value)} value={String(opt.value)}>{opt.label}</option>
                          ))}
                        </select>
                      ) : (
                        <HyperparameterInput
                          id={`${fieldId}-param-${param.name}`}
                          type={param.type}
                          value={config.hyperparameters[param.name] ?? param.default}
                          onChange={(val) => onChange({
                            ...config,
                            hyperparameters: { ...config.hyperparameters, [param.name]: val }
                          })}
                          step={param.step}
                          min={param.min}
                          max={param.max}
                        />
                      )}
                    </div>
                  ))
              )}
              {hyperparameters.length === 0 && !isLoadingHyperparamDefs && (
                 <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
                   No parameters available.
                 </p>
              )}
            </div>
        )}
    </div>
  );
}

type SearchSpaceSectionProps = Pick<TrainingSettingsState, 'config' | 'onChange' | 'searchSpaceDefs' | 'isLoadingSearchSpaceDefs'>;

export function SearchSpaceSection({
  config,
  onChange,
  searchSpaceDefs,
  isLoadingSearchSpaceDefs,
}: SearchSpaceSectionProps) {
  return (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-purple-500" />
                Hyperparameters
            </h4>
        </div>

        {isLoadingSearchSpaceDefs ? (
            <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
            </div>
        ) : (
            <div className="space-y-3">
                {searchSpaceDefs.filter(def => def.tunable !== false && isSearchSpaceParamVisible(def, config.search_space)).map(def => (
                    <div key={`${config.model_type}-${def.name}`} className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                        <SearchSpaceInput
                            def={def}
                            value={(config.search_space?.[def.name] || []) as unknown[]}
                            onChange={(newValues) => {
                                onChange({
                                    ...config,
                                    search_space: {
                                        ...config.search_space,
                                        [def.name]: newValues
                                    }
                                });
                            }}
                        />
                        {def.depends_on && (
                            <p className="mt-1 text-[10px] text-gray-400 italic">
                                Shown because &quot;{def.depends_on.param}&quot; search space includes &quot;{String(def.depends_on.value)}&quot;.
                            </p>
                        )}
                    </div>
                ))}
                {searchSpaceDefs.length === 0 && (
                    <div className="text-center py-8 text-gray-500 text-sm">
                        No hyperparameters available for this model.
                    </div>
                )}
            </div>
        )}
    </div>
  );
}
