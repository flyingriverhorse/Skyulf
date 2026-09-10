import React from 'react';
/** Renders a compact summary of active strategy params, or a "Using defaults" hint when none are set. */
export const StrategyParamsHint: React.FC<{
    strategy: string;
    strategyParams: Record<string, unknown> | undefined;
    onCustomize: () => void;
}> = ({ strategy, strategyParams, onCustomize }) => {
    const hasParams = strategyParams != null && Object.keys(strategyParams).length > 0;
    if (hasParams) {
        const parts = strategySummary(strategy, strategyParams!);
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

function strategySummary(strategy: string, params: Record<string, unknown>): string[] {
    if (strategy === 'optuna') {
        const parts = [`sampler: ${params.sampler ?? 'tpe'}`, `pruner: ${params.pruner ?? 'median'}`];
        if (params.timeout) parts.push(`timeout: ${params.timeout}s`);
        return parts;
    }
    const parts: string[] = [];
    if (params.factor) parts.push(`factor: ${params.factor}`);
    if (params.min_resources) parts.push(`min: ${params.min_resources}`);
    return parts;
}
