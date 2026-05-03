import React, { useState, useEffect } from 'react';
import { Save, RotateCcw, HelpCircle, AlertTriangle } from 'lucide-react';
import { ModalShell } from '../../../../components/shared';

// Models whose search spaces contain string/boolean/None params that CMA-ES
// cannot optimize natively — it falls back to random sampling for those params.
const CMAES_PARTIAL_MODELS = new Set([
    'logistic_regression',
    'random_forest_classifier',
    'random_forest_regressor',
    'ridge_regression',
    'linear_regression',
    'svc',
    'svr',
    'k_neighbors_classifier',
    'k_neighbors_regressor',
    'decision_tree_classifier',
    'decision_tree_regressor',
    'extra_trees_classifier',   // criterion (str), bootstrap (bool) in search space
    'extra_trees_regressor',    // criterion (str), bootstrap (bool) in search space
    'lgbm_classifier',          // boosting_type (str) in search space
    'lgbm_regressor',           // boosting_type (str) in search space
]);

export interface StrategyConfig {
    // Halving
    factor?: number;
    min_resources?: string | number;
    resource?: string;

    // Optuna
    pruner?: 'median' | 'hyperband' | 'none';
    sampler?: 'tpe' | 'random' | 'cmaes';
    timeout?: number | '';
}

interface StrategySettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (config: StrategyConfig) => void;
    strategy: string;
    initialConfig?: StrategyConfig | undefined;
    modelKey?: string | undefined;
}

const Tooltip: React.FC<{ text: string }> = ({ text }) => (
    <div className="group relative flex items-center">
        <HelpCircle className="w-3.5 h-3.5 text-gray-400 cursor-help" />
        <div className="absolute top-full mt-2 left-0 hidden group-hover:block w-56 p-2.5 bg-gray-900 text-white text-xs rounded-md shadow-xl z-[200]">
            {text}
            <div className="absolute bottom-full left-1 border-4 border-transparent border-b-gray-900" />
        </div>
    </div>
);

export const StrategySettingsModal: React.FC<StrategySettingsModalProps> = ({
    isOpen,
    onClose,
    onSave,
    strategy,
    initialConfig,
    modelKey,
}) => {
    const showCmaesWarning =
        strategy === 'optuna' &&
        modelKey !== undefined &&
        CMAES_PARTIAL_MODELS.has(modelKey);
    const isHalving = strategy === 'halving_grid' || strategy === 'halving_random';
    const isOptuna = strategy === 'optuna';

    const defaultHalving: StrategyConfig = { factor: 3, min_resources: 'exhaust', resource: 'n_samples' };
    const defaultOptuna: StrategyConfig = { pruner: 'median', sampler: 'tpe', timeout: '' };

    const [config, setConfig] = useState<StrategyConfig>({});

    useEffect(() => {
        if (isOpen) {
            if (initialConfig && Object.keys(initialConfig).length > 0) {
                setConfig(initialConfig);
            } else {
                setConfig(isHalving ? defaultHalving : defaultOptuna);
            }
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen, strategy, initialConfig]);

    if (!isHalving && !isOptuna) return null;

    const handleSave = () => {
        const finalConfig = { ...config };
        if (isOptuna && (finalConfig.timeout === '' || finalConfig.timeout === undefined)) {
            delete finalConfig.timeout;
        } else if (isOptuna && typeof finalConfig.timeout === 'string') {
            finalConfig.timeout = parseInt(finalConfig.timeout, 10);
        }
        
        onSave(finalConfig);
        onClose();
    };

    const handleReset = () => {
        setConfig(isHalving ? defaultHalving : defaultOptuna);
    };

    return (
        <ModalShell
            isOpen={isOpen}
            onClose={onClose}
            zIndex="z-[100]"
            size="md"
            title={isHalving ? 'Successive Halving Settings' : 'Optuna Settings'}
            footer={
                <div className="flex justify-between gap-3">
                    <button
                        onClick={handleReset}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition"
                    >
                        <RotateCcw size={16} />
                        Reset
                    </button>

                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition shadow-sm"
                    >
                        <Save size={16} />
                        Apply Settings
                    </button>
                </div>
            }
        >
            <div className="p-5 space-y-4">
                    {isHalving && (
                        <>
                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Factor
                                    </span>
                                    <Tooltip text="The rate at which candidate combinations are reduced in each iteration. For example, a factor of 3 means only the best 1/3 of candidates survive to the next round." />
                                </div>
                                <input
                                    type="number"
                                    min="2"
                                    value={config.factor ?? 3}
                                    onChange={(e) => setConfig({ ...config, factor: parseInt(e.target.value, 10) })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                />
                                <p className="text-xs text-gray-500 mt-1">Typical values are 2 or 3.</p>
                            </div>

                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Min Resources
                                    </span>
                                    <Tooltip text="'exhaust' automatically calculates the maximum resources to use. Alternatively, input an integer like 10 or 100 to set the starting budget explicitly." />
                                </div>
                                <input
                                    type="text"
                                    value={config.min_resources ?? 'exhaust'}
                                    onChange={(e) => setConfig({ ...config, min_resources: e.target.value })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                    placeholder="e.g., 'exhaust', 'smallest', or an integer"
                                />
                                <p className="text-xs text-gray-500 mt-1">Starting budget for the first iteration.</p>
                            </div>

                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Resource
                                    </span>
                                    <Tooltip text="Determines how resources are limited during fast initial rounds. 'n_samples' limits the amount of training data used, while 'n_estimators' limits the number of trees in ensemble models." />
                                </div>
                                <select
                                    value={config.resource ?? 'n_samples'}
                                    onChange={(e) => setConfig({ ...config, resource: e.target.value })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="n_samples">n_samples (Rows of Data)</option>
                                    <option value="n_estimators">n_estimators (Trees)</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">What defines the training budget.</p>
                            </div>
                        </>
                    )}

                    {isOptuna && (
                        <>
                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Sampler
                                    </span>
                                    <Tooltip text="TPE provides smart Bayesian learning based on past trials. Random is purely blind chance. CMA-ES is advanced evolutionary sampling meant for complex continuous search spaces." />
                                </div>
                                <select
                                    value={config.sampler ?? 'tpe'}
                                    onChange={(e) => setConfig({ ...config, sampler: e.target.value as 'tpe' | 'random' | 'cmaes' })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="tpe">TPE (Bayesian Optimization)</option>
                                    <option value="random">Random Sampler</option>
                                    <option value="cmaes">CMA-ES</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">Algorithm to suggest new parameters.</p>
                                {showCmaesWarning && config.sampler === 'cmaes' && (
                                    <div className="mt-2 flex items-start gap-2 p-2.5 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg text-xs text-amber-700 dark:text-amber-400">
                                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                                        <span>
                                            <strong>Partial CMA-ES coverage.</strong> This model has string or boolean parameters (e.g. solver, kernel, criterion) that CMA-ES cannot optimize — they will be sampled randomly. For full CMA-ES benefit, use <strong>XGBoost</strong> or <strong>Gradient Boosting</strong>. Alternatively, switch to <strong>TPE</strong> which handles mixed spaces natively.
                                        </span>
                                    </div>
                                )}
                            </div>

                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Pruner
                                    </span>
                                    <Tooltip text="Median prunes trials worse than the median of previous runs. Hyperband is an aggressively fast early-stopping algorithm. None disables early stopping." />
                                </div>
                                <select
                                    value={config.pruner ?? 'median'}
                                    onChange={(e) => setConfig({ ...config, pruner: e.target.value as 'median' | 'hyperband' | 'none' })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                >
                                    <option value="median">Median Pruner</option>
                                    <option value="hyperband">Hyperband Pruner</option>
                                    <option value="none">None (No Pruning)</option>
                                </select>
                                <p className="text-xs text-gray-500 mt-1">Algorithm to kill bad trials early.</p>
                            </div>

                            <div>
                                <div className="flex items-center gap-1.5 mb-1">
                                    <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                        Timeout (Seconds)
                                    </span>
                                    <Tooltip text="Set a hard time limit for the entire Optuna study. Regardless of N-Trials, optimization will yield the best found parameters once time is up." />
                                </div>
                                <input
                                    type="number"
                                    min="1"
                                    value={config.timeout ?? ''}
                                    onChange={(e) => setConfig({ ...config, timeout: e.target.value ? parseInt(e.target.value, 10) : '' })}
                                    className="w-full text-sm border-gray-300 dark:border-gray-600 rounded-lg p-2.5 bg-gray-50 dark:bg-gray-900 dark:text-white border focus:ring-2 focus:ring-blue-500"
                                    placeholder="e.g. 300 (Optional)"
                                />
                                <p className="text-xs text-gray-500 mt-1">Stop tuning after X seconds.</p>
                            </div>
                        </>
                    )}
                </div>
        </ModalShell>
    );
};
