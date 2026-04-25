import React, { useEffect, useState, useCallback } from 'react';
import { Activity, Loader2, Database, AlertCircle, RefreshCw, Check, X } from 'lucide-react';
import { jobsApi, JobInfo } from '../../../../core/api/jobs';
import { RegistryItem } from '../../../../core/api/registry';
import { formatMetricName } from '../../../../core/utils/format';

export interface BestParamsModalProps {
    isOpen: boolean;
    onClose: () => void;
    /** Initial model_type whose history to load. */
    modelType: string;
    /** Models offered in the dropdown selector. */
    availableModels?: RegistryItem[];
    /** When set, each row shows an "Apply" button; omit for read-only view. */
    onSelect?: (payload: { params: unknown; modelType: string }) => void;
    /** Color theme; matches the parent settings panel (Basic = blue, Advanced = purple). */
    theme?: 'blue' | 'purple';
}

/**
 * Shared "Best Parameters History" modal used by both BasicTrainingSettings
 * and AdvancedTuningSettings. Theme colors and the optional `onSelect` Apply
 * button are the only differences between the two callers.
 */
export const BestParamsModal: React.FC<BestParamsModalProps> = ({
    isOpen,
    onClose,
    modelType: initialModelType,
    availableModels = [],
    onSelect,
    theme = 'blue',
}) => {
    const [currentModelType, setCurrentModelType] = useState(initialModelType);
    const [jobs, setJobs] = useState<JobInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const colors = theme === 'purple'
        ? { iconBg: 'bg-purple-100 dark:bg-purple-900/30', iconText: 'text-purple-600 dark:text-purple-400', spinner: 'text-purple-500', ring: 'focus:ring-purple-500', hoverBorder: 'hover:border-purple-200 dark:hover:border-purple-800' }
        : { iconBg: 'bg-blue-100 dark:bg-blue-900/30',     iconText: 'text-blue-600 dark:text-blue-400',     spinner: 'text-blue-500',     ring: 'focus:ring-blue-500',     hoverBorder: 'hover:border-blue-200 dark:hover:border-blue-800' };

    useEffect(() => {
        if (isOpen) setCurrentModelType(initialModelType);
    }, [isOpen, initialModelType]);

    const fetchJobs = useCallback(() => {
        if (!currentModelType) return;
        setIsLoading(true);
        setError(null);
        jobsApi.getTuningHistory(currentModelType)
            .then(data => { setJobs(data); })
            .catch(err => {
                console.error("Failed to fetch jobs", err);
                setError("Failed to load history.");
            })
            .finally(() => { setIsLoading(false); });
    }, [currentModelType]);

    useEffect(() => {
        if (isOpen) fetchJobs();
    }, [isOpen, fetchJobs]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex justify-center items-center p-4">
            {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events,jsx-a11y/no-static-element-interactions -- backdrop dismiss zone */}
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
            <div className="relative w-full max-w-2xl max-h-[85vh] bg-white dark:bg-gray-800 shadow-2xl rounded-xl flex flex-col border border-gray-200 dark:border-gray-700 overflow-hidden animate-in fade-in zoom-in duration-200">
                <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50/50 dark:bg-gray-800/50">
                    <div className="flex items-center gap-3">
                        <div className={`p-2 ${colors.iconBg} rounded-lg`}>
                            <Activity className={`w-5 h-5 ${colors.iconText}`} />
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Best Parameters History</h3>
                            <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                    {onSelect ? 'Select parameters for:' : 'View parameters for:'}
                                </span>
                                <select
                                    value={currentModelType}
                                    onChange={(e) => { setCurrentModelType(e.target.value); }}
                                    className={`text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-0.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-1 ${colors.ring} outline-none`}
                                >
                                    {availableModels.length > 0 ? (
                                        availableModels.map(model => (
                                            <option key={model.id} value={model.id}>{model.name}</option>
                                        ))
                                    ) : (
                                        <>
                                            <option value="random_forest_classifier">Random Forest Classifier</option>
                                            <option value="logistic_regression">Logistic Regression</option>
                                            <option value="ridge_regression">Ridge Regression</option>
                                            <option value="random_forest_regressor">Random Forest Regressor</option>
                                        </>
                                    )}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={fetchJobs}
                            className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 transition-colors"
                            title="Refresh"
                        >
                            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        </button>
                        <button onClick={onClose} className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 transition-colors">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50/30 dark:bg-gray-900/30">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                            <Loader2 className={`w-8 h-8 animate-spin mb-2 ${colors.spinner}`} />
                            <p className="text-sm">Loading history...</p>
                        </div>
                    ) : error ? (
                        <div className="text-center py-12 text-red-500 bg-red-50 dark:bg-red-900/10 rounded-lg border border-red-100 dark:border-red-900/20">
                            <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                            <p className="text-sm">{error}</p>
                            <button onClick={fetchJobs} className="mt-2 text-xs underline hover:text-red-600">Try Again</button>
                        </div>
                    ) : jobs.length === 0 ? (
                        <div className="text-center py-12 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-700">
                            <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                            <p className="text-sm">No completed optimization jobs found.</p>
                            <p className="text-xs opacity-70 mt-1">Run an optimization job to see results here.</p>
                        </div>
                    ) : (
                        jobs.map(job => (
                            <div key={job.job_id} className={`group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 hover:shadow-md ${colors.hoverBorder} transition-all`}>
                                <div className="flex justify-between items-start mb-3">
                                    <div className="space-y-1">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs font-mono bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded text-gray-600 dark:text-gray-300">
                                                #{job.job_id.slice(0, 8)}
                                            </span>
                                            <span className="text-xs text-gray-400">
                                                {job.end_time ? new Date(job.end_time).toLocaleString() : 'Unknown Date'}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-3 mt-1">
                                            {typeof job.result?.best_score === 'number' && (
                                                <div className="flex items-center gap-1">
                                                    <span className="text-xs font-medium text-gray-500">
                                                        {formatMetricName((job.result as Record<string, unknown>).scoring_metric as string) || 'Score'}:
                                                    </span>
                                                    <span className="text-sm font-bold text-green-600 dark:text-green-400">
                                                        {job.result.best_score.toFixed(4)}
                                                    </span>
                                                </div>
                                            )}
                                            <span className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full flex items-center gap-1">
                                                <Check className="w-3 h-3" /> Model Ready
                                            </span>
                                        </div>
                                    </div>
                                    {onSelect && job.result?.best_params != null && (
                                        <button
                                            onClick={() => {
                                                onSelect({ params: job.result!.best_params, modelType: currentModelType });
                                                onClose();
                                            }}
                                            className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md transition-colors flex items-center gap-1 shadow-sm shadow-blue-500/20"
                                        >
                                            <Check className="w-3 h-3" />
                                            Apply
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};
