import {
    Sparkles,
    X
} from 'lucide-react';
import type { SavedThresholdInfo } from '../../../core/api/thresholdTuning';
import type { InferenceController } from './useInferenceController';

/** Edit per-request thresholds while keeping saved threshold provenance visible. */
export function ThresholdOverrides({ controller }: {
    controller: Pick<InferenceController,
        | 'overrideThresholdsEnabled' | 'setOverrideThresholdsEnabled'
        | 'overrideThresholdsValue' | 'handleOverrideThresholdChange'
        | 'handleRemoveOverrideEntry' | 'newOverrideClass' | 'setNewOverrideClass'
        | 'newOverrideThreshold' | 'setNewOverrideThreshold' | 'handleAddOverrideEntry'
        | 'singleProbMap' | 'handlePrefillFromLastPrediction' | 'savedThresholds'
        | 'activeDeployment' | 'handlePrefillFromSavedThresholds'
    >
}) {
    const {
        overrideThresholdsEnabled, setOverrideThresholdsEnabled, overrideThresholdsValue,
        handleOverrideThresholdChange, handleRemoveOverrideEntry, newOverrideClass,
        setNewOverrideClass, newOverrideThreshold, setNewOverrideThreshold,
        handleAddOverrideEntry, singleProbMap, handlePrefillFromLastPrediction,
    } = controller;
    return (<details className="mt-3 shrink-0 rounded-lg border border-gray-200 dark:border-gray-700 group overflow-hidden">
        <summary className="cursor-pointer select-none px-3 py-2 text-xs font-medium text-gray-600 dark:text-gray-300 flex items-center gap-2">
            <Sparkles className="w-3.5 h-3.5" /> Advanced: override thresholds
        </summary>
        <div className="px-3 pb-3 pt-1 space-y-2 border-t border-gray-100 dark:border-gray-700">
            <SavedThresholdDetails controller={controller} />

            <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
                <input
                    type="checkbox"
                    checked={overrideThresholdsEnabled}
                    onChange={e => setOverrideThresholdsEnabled(e.target.checked)}
                />
                Apply these thresholds to this prediction
            </label>

            {Object.entries(overrideThresholdsValue).length > 0 && (
                <div className="space-y-1">
                    {Object.entries(overrideThresholdsValue).map(([cls, thr]) => (
                        <div key={cls} className="flex items-center gap-2 flex-wrap">
                            <span
                                className="font-mono text-xs w-28 truncate text-gray-700 dark:text-gray-200"
                                title={cls}
                            >
                                {cls}
                            </span>
                            <input
                                type="number"
                                min={0}
                                max={1}
                                step={0.01}
                                value={thr}
                                onChange={e =>
                                    handleOverrideThresholdChange(cls, e.target.value)
                                }
                                className="w-24 px-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                            />
                            <button
                                onClick={() => handleRemoveOverrideEntry(cls)}
                                title="Remove"
                                className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30"
                            >
                                <X className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <div className="flex items-center gap-2 flex-wrap">
                <input
                    type="text"
                    value={newOverrideClass}
                    onChange={e => setNewOverrideClass(e.target.value)}
                    placeholder="class label"
                    className="w-28 px-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                />
                <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={newOverrideThreshold}
                    onChange={e => setNewOverrideThreshold(e.target.value)}
                    className="w-24 px-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
                />
                <button
                    onClick={handleAddOverrideEntry}
                    disabled={!newOverrideClass.trim()}
                    className="px-2 py-1 text-xs rounded border border-gray-200 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    Add
                </button>
            </div>

            {singleProbMap && (
                <button
                    onClick={handlePrefillFromLastPrediction}
                    className="text-[11px] text-blue-600 dark:text-blue-400 hover:underline"
                >
                    Use classes from last prediction
                </button>
            )}

            <p className="text-[10px] text-gray-400 italic">
                Class labels must exactly match the deployed model&apos;s classes
                (e.g. 0, 1, 2), or the prediction will be rejected.
            </p>
        </div>
    </details>);
}

function SavedThresholdDetails({ controller }: { controller: Pick<InferenceController, 'savedThresholds' | 'activeDeployment' | 'handlePrefillFromSavedThresholds'> }) {
    const { savedThresholds, activeDeployment, handlePrefillFromSavedThresholds } = controller;
    if (!savedThresholds?.thresholds || !activeDeployment) return null;
    return (
        <div
            className={`p-2 rounded border text-[11px] ${savedThresholds.enabled
                    ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800 text-blue-700 dark:text-blue-300'
                    : 'bg-slate-50 dark:bg-slate-800/60 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300'
                }`}
        >
            <div className="flex flex-wrap items-center gap-1 mb-1">
                <strong>
                    {savedThresholds.enabled
                        ? 'Saved tuned thresholds are enabled'
                        : 'Saved tuned thresholds are available but disabled'}
                </strong>
                <span className="font-mono opacity-80">
                    · Job {activeDeployment.job_id} · {activeDeployment.model_type}
                </span>
            </div>
            <SavedThresholdProvenance savedThresholds={savedThresholds} />
            <div className="flex flex-wrap gap-1 mb-1.5">
                {Object.entries(savedThresholds.thresholds).map(([cls, thr]) => (
                    <span
                        key={cls}
                        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded font-mono ${savedThresholds.enabled
                                ? 'bg-blue-100 dark:bg-blue-900/40'
                                : 'bg-slate-100 dark:bg-slate-700/80'
                            }`}
                    >
                        {cls}: {thr}
                    </span>
                ))}
            </div>
            <button
                onClick={handlePrefillFromSavedThresholds}
                className="text-current hover:underline font-medium"
            >
                Copy into override editor to tweak
            </button>
        </div>
    );
}

function SavedThresholdProvenance({ savedThresholds }: { savedThresholds: SavedThresholdInfo }) {
    return (<div className="grid grid-cols-1 gap-0.5 mb-1.5">
        <span>
            Optimized for <strong>{savedThresholds.metric ?? 'unknown metric'}</strong>
        </span>
        <span>
            Computed from <strong>{savedThresholds.split_used ?? 'unknown'}</strong> split
        </span>
        {savedThresholds.source === 'training' && (
            <span>
                Seeded at <strong>training time</strong> (Tune decision threshold)
            </span>
        )}
        {savedThresholds.computed_at && (
            <span>
                Computed at <strong>{new Date(savedThresholds.computed_at).toLocaleString()}</strong>
            </span>
        )}
    </div>);
}
