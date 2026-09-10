import { AlertCircle, ChevronDown, Loader2, Settings2 } from 'lucide-react';
import { ValidationField } from '../../../../components/shared/ValidationField';
import type { RegistryItem } from '../../../../core/api/registry';
import { HelpTooltip } from '../components/HelpTooltip';
import { HyperparameterInput } from '../components/HyperparameterInput';
import type { HyperparameterDef } from '../components/types';
import type { SegmentationConfig } from '../SegmentationSettings';

interface SettingsFieldsProps {
  config: SegmentationConfig;
  onChange: (config: SegmentationConfig) => void;
  fieldId: string;
}

/** Render model and reference controls while their disclosure state stays in the owner. */
export function SegmentationModelSection({
  config, onChange, fieldId, availableModels, isLoadingModels, requiresScaling,
  showScalingAlert, setShowScalingAlert, availableColumns, changeModel,
}: SettingsFieldsProps & {
  availableModels: RegistryItem[];
  isLoadingModels: boolean;
  requiresScaling: boolean | undefined;
  showScalingAlert: boolean;
  setShowScalingAlert: (show: boolean) => void;
  availableColumns: { name: string }[];
  changeModel: (modelType: string) => void;
}) {
  return (
    <div className="space-y-5 animate-in fade-in duration-300">
      <div className="space-y-4">
        <div className="space-y-1.5">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</span>
          <div className="grid gap-3">
            <ValidationField field="model_type">
              <label htmlFor={`${fieldId}-model_type`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Clustering Algorithm</label>
              <div className="relative">
                <select
                  id={`${fieldId}-model_type`}
                  value={config.model_type}
                  onChange={(e) => changeModel(e.target.value)}
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

            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <label htmlFor={`${fieldId}-reference_column`} className="block text-xs font-medium text-gray-700 dark:text-gray-300">Reference Column (optional)</label>
                <HelpTooltip text="A column with a known real-world label (e.g. a species/customer-type name) that you want excluded from clustering, but kept around afterward to see which cluster corresponds to which group — e.g. 'Cluster 0 is 92% setosa'. The model never sees this column." />
              </div>
              <div className="relative">
                <select
                  id={`${fieldId}-reference_column`}
                  value={config.reference_column ?? ''}
                  onChange={(e) => onChange({ ...config, reference_column: e.target.value || undefined })}
                  className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                  disabled={availableColumns.length === 0}
                >
                  <option value="">None</option>
                  {availableColumns.map((col) => (
                    <option key={col.name} value={col.name}>{col.name}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Render the parameter list without owning request or customization state. */
export function SegmentationParametersSection({
  config, onChange, fieldId, hyperparameters, isLoadingDefs,
}: SettingsFieldsProps & { hyperparameters: HyperparameterDef[]; isLoadingDefs: boolean }) {
  return (
    <div className="space-y-4 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-blue-500" />
          Hyperparameters
        </h4>
      </div>

      <div className="space-y-3">
        {isLoadingDefs ? (
          <div className="flex justify-center py-4">
            <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
          </div>
        ) : (
          hyperparameters.map((param) => (
            <SegmentationParameter key={param.name} param={param} config={config} onChange={onChange} fieldId={fieldId} />
          ))
        )}
        {hyperparameters.length === 0 && !isLoadingDefs && (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
            No parameters available.
          </p>
        )}
      </div>
    </div>
  );
}

/** Keep each dynamic control's label, fallback value and edit semantics together. */
function SegmentationParameter({ param, config, onChange, fieldId }: SettingsFieldsProps & { param: HyperparameterDef }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
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
  );
}
