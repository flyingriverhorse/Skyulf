import { Settings2 } from 'lucide-react';
import { HelpTooltip } from '../components/HelpTooltip';
import { StrategyParamsHint } from './StrategyParamsHint';
import type { TrainingSettingsState } from './useTrainingSettings';

type TuningStrategySectionProps = Pick<TrainingSettingsState,
  'config'
  | 'onChange'
  | 'fieldId'
  | 'isClassification'
  | 'setShowStrategyModal'>;

export function TuningStrategySection({
  config,
  onChange,
  fieldId,
  isClassification,
  setShowStrategyModal,
}: TuningStrategySectionProps) {
  return (
    <>
        <div className="border-t border-gray-100 dark:border-gray-700" />
        {/* Strategy & Metrics */}
        <div className="space-y-1.5">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tuning Strategy</span>
            <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2">
                  <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1.5">
                            <label htmlFor={`${fieldId}-search-method`} className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                Search Method
                            </label>
                            <HelpTooltip placement="bottom-left" text={searchStrategyHelp(config.search_strategy)} />
                        </div>
                      {hasStrategySettings(config.search_strategy) && (
                          <button
                              type="button"
                              onClick={() => setShowStrategyModal(true)}
                              className="text-blue-600 hover:text-blue-700 dark:text-blue-400 p-1 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 transition group flex items-center justify-center"
                              aria-label="Search strategy settings"
                              title={`${config.search_strategy.replace('_', ' ')} Settings`}
                          >
                              <Settings2 size={14} className="group-hover:rotate-45 transition-transform duration-300" />
                          </button>
                      )}
                  </div>
                  <select
                      id={`${fieldId}-search-method`}
                      value={config.search_strategy ?? 'random'}
                      onChange={(e) => {
                          const newStrategy = e.target.value;
                          // Always clear strategy_params when changing strategy — prevents stale
                          // optuna/halving settings (e.g. sampler=cmaes) from silently carrying
                          // over to a different strategy or a fresh selection.
                          onChange({
                              ...config,
                              search_strategy: newStrategy,
                              strategy_params: {}
                          });
                      }}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    >
                        <option value="random">Random Search</option>
                        <option value="grid">Grid Search</option>
                        <option value="halving_grid">Successive Halving (Grid)</option>
                        <option value="halving_random">Successive Halving (Randomized)</option>
                        <option value="optuna">Optuna Search</option>
                    </select>
                  {/* Show active settings summary — configured params or default hint */}
                  {hasStrategySettings(config.search_strategy) && (
                      <StrategyParamsHint
                          strategy={config.search_strategy}
                          strategyParams={config.strategy_params}
                          onCustomize={() => setShowStrategyModal(true)}
                      />
                  )}
                </div>

                <div>
                    <label htmlFor={`${fieldId}-metric`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Metric</label>
                    <select
                        id={`${fieldId}-metric`}
                        value={config.metric}
                        onChange={(e) => onChange({ ...config, metric: e.target.value })}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                    >
                        <option value="accuracy">Accuracy</option>
                        <option value="f1">F1 Score</option>
                        <option value="roc_auc">ROC AUC</option>
                        <option value="mse">MSE</option>
                        <option value="rmse">RMSE</option>
                        <option value="mae">MAE</option>
                        <option value="r2">R2 Score</option>
                    </select>
                </div>

                <div>
                    <label htmlFor={`${fieldId}-trials`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Trials</label>
                    <input
                        id={`${fieldId}-trials`}
                        type="number"
                        value={config.n_trials}
                        onChange={(e) => onChange({ ...config, n_trials: Number(e.target.value) })}
                        disabled={['grid', 'halving_grid'].includes(config.search_strategy)}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 disabled:opacity-50"
                        min={1}
                    />
                </div>

                <div>
                    <div className="flex items-center gap-1.5 mb-1">
                        <label htmlFor={`${fieldId}-random-state`} className="block text-xs font-medium text-gray-700 dark:text-gray-300">Random State</label>
                        <HelpTooltip text="Seed for the search and the final refit — same seed + same data = identical tuning outcome. Applies to every candidate, not just the winner." />
                    </div>
                    <input
                        id={`${fieldId}-random-state`}
                        type="number"
                        value={config.random_state ?? 42}
                        onChange={(e) => onChange({ ...config, random_state: Number(e.target.value) })}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                        min={0}
                    />
                </div>

                {isClassification && (
                    <div className="col-span-2 flex items-center gap-2">
                        <input
                            type="checkbox"
                            id={`${fieldId}-tune_threshold`}
                            checked={config.tune_threshold ?? false}
                            onChange={(e) => onChange({ ...config, tune_threshold: e.target.checked })}
                            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor={`${fieldId}-tune_threshold`} className="text-xs text-gray-600 dark:text-gray-400">
                            Tune decision threshold
                        </label>
                        <HelpTooltip text="After tuning, picks the probability cutoff that maximises your metric on the validation split (binary targets). Predictions then use that cutoff instead of the default 0.5. Needs a validation split; probability-only metrics like ROC AUC fall back to balanced accuracy for the cutoff search." />
                    </div>
                )}
            </div>
        </div>
    </>
  );
}


function hasStrategySettings(strategy: string): boolean {
    return ['halving_grid', 'halving_random', 'optuna'].includes(strategy);
}

function searchStrategyHelp(strategy: string): string {
    switch (strategy) {
        case 'optuna': return 'Optuna uses Bayesian optimization (TPE) to efficiently find optimal hyperparameters with early pruning.';
        case 'halving_grid': return 'Successive Halving (Grid) tests all combinations but quickly drops poorly performing candidates to save time.';
        case 'halving_random': return 'Successive Halving (Random) tests random combinations but quickly drops poorly performing candidates.';
        case 'grid': return 'Grid Search tests every single combination in the search space. Can be very slow and computationally expensive.';
        default: return 'Random Search tests a random subset of parameter combinations. Fast and often surprisingly effective compared to Grid Search.';
    }
}
