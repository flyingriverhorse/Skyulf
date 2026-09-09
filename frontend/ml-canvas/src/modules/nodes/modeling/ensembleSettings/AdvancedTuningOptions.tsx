import { useId, useState } from 'react';
import { ChevronRight, Settings2, Sparkles } from 'lucide-react';
import type { EnsembleConfig } from '../EnsembleSettings';
import { HelpTooltip } from '../components/HelpTooltip';
import { StrategySettingsModal } from '../components/StrategySettingsModal';
import { metricOptions, SEARCH_STRATEGIES, type UpdateFn } from './modelOptions';

/** Tuning controls shown in Advanced mode: search strategy, trial budget, metric. */
export function AdvancedTuningOptions({ config, update }: { config: EnsembleConfig; update: UpdateFn }) {
  const fieldId = useId();
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [showBaseTuningDetails, setShowBaseTuningDetails] = useState(false);
  const showStrategyBtn = ['halving_grid', 'halving_random', 'optuna'].includes(config.search_strategy);

  return (
    <div className="space-y-3 p-3 rounded-lg border border-purple-100 dark:border-purple-800 bg-purple-50/50 dark:bg-purple-900/10">
      <div className="flex items-center gap-1.5">
        <Sparkles className="w-3.5 h-3.5 text-purple-500" />
        <span className="text-xs font-semibold text-purple-700 dark:text-purple-300">Hyperparameter Tuning</span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="flex items-center gap-1 text-xs font-medium text-gray-700 dark:text-gray-300">
              <label htmlFor={`${fieldId}-search_strategy`}>Search Strategy</label>
              <HelpTooltip
                placement="bottom-left"
                text="Searches the ensemble's own params (e.g. voting/cv), not each base model. Enable &quot;Tune base model hyperparameters&quot; below to also search each model's params."
              />
            </span>
            {showStrategyBtn && (
              <button
                type="button"
                onClick={() => { setShowStrategyModal(true); }}
                className="text-blue-600 hover:text-blue-700 dark:text-blue-400 p-0.5 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 transition group flex items-center justify-center"
                aria-label="Search strategy settings"
                title={`${config.search_strategy.replace('_', ' ')} Settings`}
              >
                <Settings2 size={13} className="group-hover:rotate-45 transition-transform duration-300" />
              </button>
            )}
          </div>
          <select
            id={`${fieldId}-search_strategy`}
            value={config.search_strategy}
            onChange={(e) => {
              update({
                search_strategy: e.target.value,
                strategy_params: {} // clear params on change to prevent carryover
              });
            }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
          >
            {SEARCH_STRATEGIES.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={`${fieldId}-n_trials`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Trials</label>
          <input
            id={`${fieldId}-n_trials`}
            type="number"
            min={1}
            value={config.n_trials}
            onChange={(e) => { update({ n_trials: Number(e.target.value) }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          />
        </div>
        <div>
          <span className="flex items-center gap-1 text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
            <label htmlFor={`${fieldId}-random_state`}>Random State</label>
            <HelpTooltip text="Seed for the search and the final refit — same seed + same data = identical tuning outcome." />
          </span>
          <input
            id={`${fieldId}-random_state`}
            type="number"
            min={0}
            value={config.random_state ?? 42}
            onChange={(e) => { update({ random_state: Number(e.target.value) }); }}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
          />
        </div>
      </div>
      <div>
        <label htmlFor={`${fieldId}-metric`} className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Optimize Metric</label>
        <select
          id={`${fieldId}-metric`}
          value={config.metric}
          onChange={(e) => { update({ metric: e.target.value }); }}
          className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 outline-none"
        >
          {metricOptions(config.task).map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      <label className="flex items-start gap-2 text-xs text-gray-700 dark:text-gray-300">
        <input
          type="checkbox"
          checked={config.tune_base_models !== false}
          onChange={(e) => { update({ tune_base_models: e.target.checked }); }}
          className="mt-0.5 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
        />
        <span className="flex-1">
          <span className="inline-flex items-center gap-1">
            <strong>Tune base model hyperparameters</strong>
            <button
              type="button"
              aria-label="Base model tuning details"
              aria-expanded={showBaseTuningDetails}
              onClick={() => { setShowBaseTuningDetails(!showBaseTuningDetails); }}
              className="text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
              title={showBaseTuningDetails ? 'Hide details' : 'Show details'}
            >
              <ChevronRight
                className={`w-3 h-3 transition-transform ${showBaseTuningDetails ? 'rotate-90' : ''}`}
              />
            </button>
          </span>
          {showBaseTuningDetails && (
            <span className="block mt-1 text-gray-500 dark:text-gray-400">
              Automatically searches each selected base learner&apos;s default parameter range
              (e.g. <code>random_forest__n_estimators</code>), in addition to the ensemble&apos;s own
              params. Ranges are picked automatically per model — uncheck this to tune only the
              ensemble-level params and keep base models at the fixed values set in Basic mode.
            </span>
          )}
        </span>
      </label>

      {showStrategyModal && (
        <StrategySettingsModal
          isOpen={showStrategyModal}
          onClose={() => { setShowStrategyModal(false); }}
          onSave={(p) => { update({ strategy_params: p as Record<string, unknown> }); }}
          strategy={config.search_strategy}
          initialConfig={config.strategy_params}
          modelKey={config.model_type}
        />
      )}
    </div>
  );
}
