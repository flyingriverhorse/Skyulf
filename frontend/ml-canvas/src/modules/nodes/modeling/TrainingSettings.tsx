import React, { useEffect, useState, useRef } from 'react';
import {
    Play, Download, Loader2, Activity, Settings2,
    BarChart3, X, ChevronRight, ChevronDown, AlertCircle, AlertTriangle
} from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import { RegistryItem, registryApi } from '../../../core/api/registry';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { HelpTooltip } from './components/HelpTooltip';
import { BestParamsModal } from './components/BestParamsModal';
import { HyperparameterInput } from './components/HyperparameterInput';
import { SearchSpaceInput } from './components/SearchSpaceInput';
import { StrategySettingsModal, StrategyConfig } from './components/StrategySettingsModal';
import type { HyperparameterDef } from './components/types';
import type { ExecutionMode } from '../../../core/types/executionMode';

export type TrainingRunMode = 'basic' | 'advanced';

export interface TrainingConfig {
  run_mode: TrainingRunMode;
  target_column: string;
  model_type: string;
  // Basic-mode fixed hyperparameters.
  hyperparameters: Record<string, unknown>;
  // Advanced-mode tuning fields.
  search_space: Record<string, unknown>;
  n_trials: number;
  metric: string;
  search_strategy: string;
  strategy_params?: Record<string, unknown>;
  random_state: number;
  // Shared CV section.
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
  cv_time_column?: string;
  execution_mode?: ExecutionMode;
}

/** Two-option segmented control matching `EnsembleSettings`'s run-mode toggle. */
const RunModeToggle: React.FC<{ value: TrainingRunMode; onSelect: (v: TrainingRunMode) => void }> = ({ value, onSelect }) => (
  <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
    {([['basic', 'Basic'], ['advanced', 'Advanced (Tuning)']] as const).map(([v, label]) => (
      <button
        key={v}
        type="button"
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

/** Renders a compact summary of active strategy params, or a "Using defaults" hint when none are set. */
const StrategyParamsHint: React.FC<{
    strategy: string;
    strategyParams: Record<string, unknown> | undefined;
    onCustomize: () => void;
}> = ({ strategy, strategyParams, onCustomize }) => {
    const hasParams = strategyParams != null && Object.keys(strategyParams).length > 0;
    if (hasParams) {
        const parts: string[] = [];
        if (strategy === 'optuna') {
            parts.push(`sampler: ${strategyParams!.sampler ?? 'tpe'}`);
            parts.push(`pruner: ${strategyParams!.pruner ?? 'median'}`);
            if (strategyParams!.timeout) parts.push(`timeout: ${strategyParams!.timeout}s`);
        } else {
            if (strategyParams!.factor) parts.push(`factor: ${strategyParams!.factor}`);
            if (strategyParams!.min_resources) parts.push(`min: ${strategyParams!.min_resources}`);
        }
        return (
            <>
                <p className="mt-1.5 text-xs text-blue-600 dark:text-blue-400">
                    ⚙ {parts.join(' · ')}
                </p>
                {strategy === 'halving_grid' && (
                    <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                        ⚠️ Evaluates the full grid — large search spaces can take many minutes. Reduce candidate values in the <strong>Search Space</strong> section below, or switch to <strong>halving_random</strong>.
                    </p>
                )}
            </>
        );
    }
    const defaultHint = strategy === 'optuna'
        ? 'sampler: tpe · pruner: median'
        : 'factor: 3 · min: exhaust';
    return (
        <>
            <p className="mt-1.5 text-xs text-gray-500 dark:text-gray-400">
                Using defaults ({defaultHint}).{' '}
                <button
                    type="button"
                    onClick={onCustomize}
                    className="underline hover:text-blue-500 transition-colors"
                >
                    Customize
                </button>
            </p>
            {strategy === 'halving_grid' && (
                <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                    ⚠️ Evaluates the full grid — large search spaces can take many minutes. Reduce candidate values in the <strong>Search Space</strong> section below, or switch to <strong>halving_random</strong>.
                </p>
            )}
        </>
    );
};

export const TrainingSettings: React.FC<{
  config: TrainingConfig;
  onChange: (c: TrainingConfig) => void;
  nodeId?: string;
}> = ({
  config,
  onChange,
  nodeId,
}) => {
  const isAdvanced = config.run_mode === 'advanced';

  // Basic-mode hyperparameter defs/editor state.
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [isLoadingHyperparamDefs, setIsLoadingHyperparamDefs] = useState(false);

  // Advanced-mode search-space defs/editor state.
  const [searchSpaceDefs, setSearchSpaceDefs] = useState<HyperparameterDef[]>([]);
  const [isLoadingSearchSpaceDefs, setIsLoadingSearchSpaceDefs] = useState(false);
  const [showStrategyModal, setShowStrategyModal] = useState(false);

  const [showParamsModal, setShowParamsModal] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_training_node'));

  const { availableColumns, upstreamTarget, datasetId, runJob } = useTrainingNodeContext(nodeId);

  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();
  const [activeTab, setActiveTab] = useState<'model' | 'params'>('model');
  const [showCV, setShowCV] = useState(false);
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showScalingAlert, setShowScalingAlert] = useState(true);

  // Find currently selected model item to check tags
  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');

  // Fetch available models from registry. Shared between both modes — the
  // dedicated Segmentation node has its own model list (see
  // `SegmentationSettings`); this node always excludes clustering algorithms,
  // since its target-column/CV-driven flow doesn't apply to them.
  useEffect(() => {
      const fetchModels = async () => {
          setIsLoadingModels(true);
          try {
              const nodes = await registryApi.getAllNodes();
              // Accept both "Model" (old) and "Modeling" (new skyulf-core).
              const models = nodes.filter(n => {
                  const isModeling = n.category === 'Model' || n.category === 'Modeling';
                  const isClustering = n.tags?.includes('clustering') ?? false;
                  return isModeling && !isClustering;
              });
              setAvailableModels(models);
          } catch (error) {
              console.error("Failed to fetch models:", error);
              // Fallback to static list if API fails
              setAvailableModels([
                  { id: 'random_forest_classifier', name: 'Random Forest Classifier', category: 'Modeling', description: '', params: {} },
                  { id: 'logistic_regression', name: 'Logistic Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'sgd_classifier', name: 'SGD Classifier', category: 'Modeling', description: '', params: {} },
                  { id: 'ridge_regression', name: 'Ridge Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'random_forest_regressor', name: 'Random Forest Regressor', category: 'Modeling', description: '', params: {} },
              ]);
          } finally {
              setIsLoadingModels(false);
          }
      };
      fetchModels();
  }, []);

  // We use a ref to track if customization was active before model switch (basic mode).
  const keepCustomizationOpen = useRef(false);

  // Basic mode: fetch hyperparameter definitions when model type changes.
  useEffect(() => {
    if (isAdvanced) return;
    if (config.model_type) {
      setIsLoadingHyperparamDefs(true);
      jobsApi.getHyperparameters(config.model_type)
        .then((defs) => {
            const definitions = defs as HyperparameterDef[];
            setHyperparameters(definitions);

            // If we switched models while customization was active, apply new defaults immediately
            if (keepCustomizationOpen.current) {
                const defaults: Record<string, unknown> = {};
                definitions.forEach(p => {
                    defaults[p.name] = p.default;
                });
                onChange({ ...config, hyperparameters: defaults });
                keepCustomizationOpen.current = false;
            }
        })
        .catch(console.error)
        .finally(() => { setIsLoadingHyperparamDefs(false); });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type, isAdvanced]);

  // Advanced mode: fetch search-space definitions/defaults when model or
  // strategy changes. Mirrors the former AdvancedTuningSettings effect.
  const loadedModelTypeRef = useRef<string | null>(null);
  const loadedStrategyRef = useRef<string | null>(null);

  useEffect(() => {
      if (!isAdvanced) return;
      const loadModelData = async () => {
          const modelType = config.model_type;
          const strategy = config.search_strategy ?? 'random';
          if (!modelType) return;

          const isNewModel = modelType !== loadedModelTypeRef.current;
          // Also reload when switching between grid and non-grid strategies
          // so the search space is appropriate for the selected method.
          const wasGrid = loadedStrategyRef.current === 'grid' || loadedStrategyRef.current === 'halving_grid';
          const isGrid = strategy === 'grid' || strategy === 'halving_grid';
          const isStrategyClassChange = wasGrid !== isGrid;

          if (!isNewModel && !isStrategyClassChange && searchSpaceDefs.length > 0) return;

          setIsLoadingSearchSpaceDefs(true);
          try {
              // 1. Fetch Definitions (only needed when model changes)
              if (isNewModel) {
                  const defs = await jobsApi.getHyperparameters(modelType);
                  setSearchSpaceDefs(defs as HyperparameterDef[]);
              }

              // 2. Fetch Defaults when model or strategy class changes
              if (isNewModel || isStrategyClassChange) {
                  const defaults = await jobsApi.getDefaultSearchSpace(modelType, strategy);
                  onChange({
                      ...config,
                      search_space: defaults || {}
                  });
                  loadedModelTypeRef.current = modelType;
                  loadedStrategyRef.current = strategy;
              }
          } catch (error) {
              console.error("Failed to load model hyperparameters:", error);
          } finally {
              setIsLoadingSearchSpaceDefs(false);
          }
      };

      void loadModelData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type, config.search_strategy, isAdvanced]);

  // Auto-select target column from upstream
  useEffect(() => {
    if (upstreamTarget && config.target_column !== upstreamTarget) {
        if (isAdvanced) {
            onChange({ ...config, target_column: upstreamTarget });
        } else if (!config.target_column) {
            // Basic mode only auto-selects when nothing is set yet — matches
            // the former BasicTrainingSettings behavior (advanced mode always
            // synced, basic mode preserved a manual choice).
            onChange({ ...config, target_column: upstreamTarget });
        }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamTarget, config.target_column, isAdvanced]);

  const handleSubmit = async () => {
    await runJob(isAdvanced ? 'advanced_tuning' : 'basic_training');
  };

  const ModelConfigSection = (
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
                    <div>
                        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Model Type</span>
                        <div className="relative">
                            <select
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
                    </div>

                    <div>
                        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</span>
                        {availableColumns.length === 0 && (
                            <div className="mb-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-xs text-yellow-700 dark:text-yellow-400 flex items-center gap-2">
                                <AlertTriangle className="w-3 h-3" />
                                <span>Connect a dataset node to see available columns</span>
                            </div>
                        )}
                        <div className="relative">
                            {availableColumns.length > 0 ? (
                                <select
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
                                    type="text"
                                    value={config.target_column}
                                    onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                                    className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                    placeholder="e.g., target"
                                />
                            )}
                            {availableColumns.length > 0 && <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />}
                        </div>
                    </div>
                </div>
            </div>

            {isAdvanced && (
                <>
                    <div className="border-t border-gray-100 dark:border-gray-700" />
                    {/* Strategy & Metrics */}
                    <div className="space-y-1.5">
                        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tuning Strategy</span>
                        <div className="grid grid-cols-2 gap-3">
                            <div className="col-span-2">
                              <div className="flex items-center justify-between mb-1">
                                    <div className="flex items-center gap-1.5">
                                        <span className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                            Search Method
                                        </span>
                                        <HelpTooltip placement="bottom-left" text={
                                            config.search_strategy === 'optuna' ? 'Optuna uses Bayesian optimization (TPE) to efficiently find optimal hyperparameters with early pruning.' :
                                            config.search_strategy === 'halving_grid' ? 'Successive Halving (Grid) tests all combinations but quickly drops poorly performing candidates to save time.' :
                                            config.search_strategy === 'halving_random' ? 'Successive Halving (Random) tests random combinations but quickly drops poorly performing candidates.' :
                                            config.search_strategy === 'grid' ? 'Grid Search tests every single combination in the search space. Can be very slow and computationally expensive.' :
                                            'Random Search tests a random subset of parameter combinations. Fast and often surprisingly effective compared to Grid Search.'
                                        } />
                                    </div>
                                  {(config.search_strategy === 'halving_grid' || config.search_strategy === 'halving_random' || config.search_strategy === 'optuna') && (
                                      <button
                                          type="button"
                                          onClick={() => setShowStrategyModal(true)}
                                          className="text-blue-600 hover:text-blue-700 dark:text-blue-400 p-1 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 transition group flex items-center justify-center"
                                          title={`${config.search_strategy.replace('_', ' ')} Settings`}
                                      >
                                          <Settings2 size={14} className="group-hover:rotate-45 transition-transform duration-300" />
                                      </button>
                                  )}
                              </div>
                              <select
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
                              {(config.search_strategy === 'optuna' || config.search_strategy === 'halving_grid' || config.search_strategy === 'halving_random') && (
                                  <StrategyParamsHint
                                      strategy={config.search_strategy}
                                      strategyParams={config.strategy_params}
                                      onCustomize={() => setShowStrategyModal(true)}
                                  />
                              )}
                            </div>

                            <div>
                                <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Metric</span>
                                <select
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
                                <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Trials</span>
                                <input
                                    type="number"
                                    value={config.n_trials}
                                    onChange={(e) => onChange({ ...config, n_trials: Number(e.target.value) })}
                                    disabled={['grid', 'halving_grid'].includes(config.search_strategy)}
                                    className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 disabled:opacity-50"
                                    min={1}
                                />
                            </div>
                        </div>
                    </div>
                </>
            )}

            <div className="border-t border-gray-100 dark:border-gray-700" />

            {/* CV Settings */}
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <button
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
                                id="cv_enabled"
                                checked={config.cv_enabled !== false}
                                onChange={(e) => onChange({ ...config, cv_enabled: e.target.checked })}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <label htmlFor="cv_enabled" className="text-sm text-gray-700 dark:text-gray-300">Enable Cross-Validation</label>
                        </div>

                        {config.cv_enabled !== false && (
                            <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <span className="block text-xs text-gray-500 mb-1">Folds</span>
                                        <input
                                            type="number"
                                            value={config.cv_folds ?? 5}
                                            onChange={(e) => onChange({ ...config, cv_folds: Number(e.target.value) })}
                                            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                            min={2}
                                        />
                                    </div>
                                    <div>
                                        <span className="block text-xs text-gray-500 mb-1">Method</span>
                                        <select
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
                                            <span className="block text-xs text-gray-500 mb-1">Time Column (optional)</span>
                                            <select
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
                                        id="cv_shuffle"
                                        checked={config.cv_shuffle !== false}
                                        onChange={(e) => onChange({ ...config, cv_shuffle: e.target.checked })}
                                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                    />
                                    <label htmlFor="cv_shuffle" className="text-xs text-gray-600 dark:text-gray-400">Shuffle Data</label>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    </div>
  );

  // Defining these as nested components caused React to remount them on every
  // parent render — losing input focus and re-running their effects. Use JSX
  // consts instead so they share this component's render tree.
  const useCustomParams = Object.keys(config.hyperparameters).length > 0;

  const toggleCustomParams = (enabled: boolean) => {
      if (enabled) {
          // Initialize with defaults if empty
          const defaults: Record<string, unknown> = {};
          hyperparameters.forEach(p => {
              defaults[p.name] = p.default;
          });
          // Only update if we don't have params already
          if (Object.keys(config.hyperparameters).length === 0) {
              onChange({ ...config, hyperparameters: defaults });
          }
      } else {
          onChange({ ...config, hyperparameters: {} });
      }
  };

  const HyperparametersSection = (
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
                  hyperparameters.map((param) => (
                    <div key={param.name} className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                      <div className="flex justify-between items-center mb-2">
                          <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                            {param.label}
                          </label>
                          {param.description && <HelpTooltip text={param.description} />}
                      </div>
                      {param.type === 'select' ? (
                        <select
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

  const SearchSpaceSection = (
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
                {searchSpaceDefs.map(def => (
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

  const SecondaryPanel = isAdvanced ? SearchSpaceSection : HyperparametersSection;
  const secondaryTabLabel = isAdvanced ? 'Search Space' : 'Hyperparameters';

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded text-xs text-blue-700 dark:text-blue-300 flex justify-between items-start gap-2">
          <span>Train a model with fixed parameters, or switch to Advanced to automatically tune it.</span>
          <button
            onClick={() => {
                setShowInfo(false);
                sessionStorage.setItem('hide_info_training_node', 'true');
            }}
            className="text-blue-400 hover:text-blue-600 dark:hover:text-blue-200"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

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

      {/* Mobile/Narrow Tabs */}
      {!isWide && (
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'model'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            onClick={() => { setActiveTab('model'); }}
          >
            Configuration
          </button>
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'params'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            onClick={() => { setActiveTab('params'); }}
          >
            {secondaryTabLabel}
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-1 pb-4 custom-scrollbar">
        {isWide ? (
            <div className="grid grid-cols-2 gap-6 h-full">
                <div className="overflow-y-auto pr-2">{ModelConfigSection}</div>
                <div className="overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-800">{SecondaryPanel}</div>
            </div>
        ) : (
            <>
                {activeTab === 'model' && ModelConfigSection}
                {activeTab === 'params' && SecondaryPanel}
            </>
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          onClick={() => { void handleSubmit(); }}
          disabled={!datasetId}
          title={!datasetId ? 'Connect a dataset node upstream to enable training' : undefined}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-lg disabled:hover:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          <Play className="w-4 h-4 fill-current" />
          <span className="text-sm font-semibold">{isAdvanced ? 'Start Advanced Training' : 'Start Training'}</span>
        </button>

        {isAdvanced && (
            <button
                onClick={() => { setShowParamsModal(true); }}
                className="text-xs text-gray-500 hover:text-purple-600 dark:text-gray-400 dark:hover:text-purple-400 flex items-center gap-1.5 transition-colors px-3 py-1 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800"
            >
                <Activity className="w-3.5 h-3.5" />
                View Best Parameters History
            </button>
        )}
      </div>
    </div>
  );
};
