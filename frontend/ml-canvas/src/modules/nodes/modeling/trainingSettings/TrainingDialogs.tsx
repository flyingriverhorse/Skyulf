import { BestParamsModal } from '../components/BestParamsModal';
import { StrategySettingsModal, StrategyConfig } from '../components/StrategySettingsModal';
import type { TrainingSettingsState } from './useTrainingSettings';

type TrainingDialogsProps = Pick<TrainingSettingsState,
  'config'
  | 'onChange'
  | 'isAdvanced'
  | 'showParamsModal'
  | 'setShowParamsModal'
  | 'availableModels'
  | 'showStrategyModal'
  | 'setShowStrategyModal'>;

export function TrainingDialogs({
  config,
  onChange,
  isAdvanced,
  showParamsModal,
  setShowParamsModal,
  availableModels,
  showStrategyModal,
  setShowStrategyModal,
}: TrainingDialogsProps) {
  return (
    <>
          <BestParamsModal
            isOpen={showParamsModal}
            onClose={() => { setShowParamsModal(false); }}
            modelType={config.model_type}
            availableModels={availableModels}
            theme={isAdvanced ? 'purple' : 'blue'}
            // Advanced mode's history modal is read-only (opened via the footer
            // link, below); only basic mode's "Load Best Params" wires an Apply
            // handler. `exactOptionalPropertyTypes` requires the prop be omitted
            // entirely rather than explicitly set to `undefined`.
            {...(!isAdvanced ? {
                onSelect: (result: { modelType: string; params: unknown }) => {
                    if (result.modelType && result.modelType !== config.model_type) {
                        onChange({
                            ...config,
                            model_type: result.modelType,
                            hyperparameters: result.params as Record<string, unknown>
                        });
                    } else {
                        onChange({ ...config, hyperparameters: result.params as Record<string, unknown> });
                    }
                },
            } : {})}
          />

          {isAdvanced && (
              <StrategySettingsModal
                  isOpen={showStrategyModal}
                  onClose={() => setShowStrategyModal(false)}
                  strategy={config.search_strategy || 'random'}
                  initialConfig={config.strategy_params as StrategyConfig | undefined}
                  modelKey={config.model_type}
                  onSave={(newStrategyParams) => {
                      onChange({ ...config, strategy_params: newStrategyParams as Record<string, unknown> });
                  }}
              />
          )}
    </>
  );
}
