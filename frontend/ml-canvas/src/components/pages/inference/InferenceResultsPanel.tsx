import {
    AlertCircle, BarChart3, CheckCircle, Copy, Download, FileSpreadsheet, History,
    LayoutGrid, List, Loader2, Play, RotateCcw, Sparkles, Trash2, XCircle, Zap,
} from 'lucide-react';
import { EmptyState, ErrorState } from '../../shared';
import {
    HISTORY_TTL_MS, MAX_RECENT_RUNS, asProbabilityMap, describeThresholdContext,
    formatTime, renderPrediction,
} from './inferenceData';
import { InputOutputTable, LatencySparkline, PredictionHistogram, ProbabilityBars } from './InferenceVisuals';
import type { InferenceController } from './useInferenceController';

/** Keep settled results visible alongside pending runs and durable run history. */
export function InferenceResultsPanel({ controller }: {
    controller: Pick<InferenceController,
        | 'singleProbMap' | 'latencyMs' | 'predictions' | 'setResultsView' | 'resultsView'
        | 'handleCopyPredictions' | 'handleDownloadJson' | 'handleDownloadCsv'
        | 'currentRunMeta' | 'activeRun' | 'predictionStats' | 'thresholdsApplied' | 'error'
        | 'parsedInputRows' | 'schemaChips' | 'lastAttemptRef' | 'handleRetryRun'
        | 'runHistory' | 'recentLatencies' | 'handleClearRunHistory' | 'handleRestoreRun'
    >
}) {
    const { singleProbMap } = controller;
    return (<div className="flex flex-col lg:h-full lg:min-h-0">
        {/* Results panel */}
        <div className="bg-white dark:bg-gray-800 p-4 sm:p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col flex-1 min-h-[20rem] lg:min-h-0">
            <ResultsToolbar controller={controller} />

            {/* Named-run provenance: which model/version, which input, when,
                            and under what threshold context — durable across reload via
                            `currentRunMeta`, not just an ephemeral in-memory flag. */}
            <RunProvenance controller={controller} />

            {/* Pending banner — kept separate from the settled-run display below so
                            an earlier result stays visible (and legible) while a new run is in
                            flight, instead of being wiped the instant a request starts. */}
            <PendingRunNotice controller={controller} />

            {/* Numeric stats strip + tiny histogram. */}
            <PredictionStatistics controller={controller} />

            {/* Probability bars for single-row classification responses. */}
            {singleProbMap && (
                <div className="mb-3 p-3 rounded bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700">
                    <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-2 flex items-center gap-1">
                        <BarChart3 className="w-3 h-3" /> Class probabilities
                    </div>
                    <ProbabilityBars probs={singleProbMap} />
                </div>
            )}

            {/* Which thresholds the backend actually applied for this run (override or deployment default). */}
            <AppliedThresholds controller={controller} />

            <ResultsBody controller={controller} />

            {/* Recent runs strip — click to restore an earlier run's exact input
                            and outcome. Durable across reload (localStorage-backed), with
                            explicit retention/expiry so it's clear this isn't a server copy. */}
            <RecentRuns controller={controller} />
        </div>
    </div>);
}

function ResultsToolbar({ controller }: {
    controller: Pick<InferenceController,
        | 'latencyMs' | 'predictions' | 'setResultsView' | 'resultsView'
        | 'handleCopyPredictions' | 'handleDownloadJson' | 'handleDownloadCsv'
    >
}) {
    const {
        latencyMs, predictions, setResultsView, resultsView, handleCopyPredictions,
        handleDownloadJson, handleDownloadCsv,
    } = controller;
    return (<div className="flex justify-between items-center mb-4 gap-3 flex-wrap">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
            Prediction Results
        </h3>
        <div className="flex items-center gap-3 flex-wrap">
            {latencyMs != null && (
                <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
                    <Zap className="w-3 h-3" /> {latencyMs} ms (client-observed)
                </span>
            )}
            {predictions && (
                <>
                    <div className="flex border border-gray-200 dark:border-gray-700 rounded overflow-hidden">
                        <button
                            onClick={() => setResultsView('list')}
                            className={`flex items-center gap-1 text-xs px-2 py-1 transition-colors ${resultsView === 'list'
                                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                                    : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                }`}
                            title="List view"
                        >
                            <List className="w-3 h-3" /> List
                        </button>
                        <button
                            onClick={() => setResultsView('table')}
                            className={`flex items-center gap-1 text-xs px-2 py-1 transition-colors border-l border-gray-200 dark:border-gray-700 ${resultsView === 'table'
                                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                                    : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                }`}
                            title="Side-by-side input/output table"
                        >
                            <LayoutGrid className="w-3 h-3" /> Table
                        </button>
                    </div>
                    <button
                        onClick={() => void handleCopyPredictions()}
                        className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        title="Copy as JSON"
                    >
                        <Copy className="w-3 h-3" /> Copy
                    </button>
                    <button
                        onClick={handleDownloadJson}
                        className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        title="Download as JSON"
                    >
                        <Download className="w-3 h-3" /> JSON
                    </button>
                    <button
                        onClick={handleDownloadCsv}
                        className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        title="Download inputs + predictions as CSV"
                    >
                        <FileSpreadsheet className="w-3 h-3" /> CSV
                    </button>
                </>
            )}
        </div>
    </div>);
}

function RunProvenance({ controller }: { controller: Pick<InferenceController, 'currentRunMeta' | 'resultsView'> }) {
    const { currentRunMeta, resultsView } = controller;
    return (<>{currentRunMeta && (
        <div
            className={`mb-3 p-2.5 rounded border text-[11px] flex flex-wrap items-center gap-x-3 gap-y-1 ${runProvenanceClass(currentRunMeta.status)
                }`}
            data-testid="run-provenance"
        >
            <strong className="font-mono">{currentRunMeta.label}</strong>
            <span>
                {currentRunMeta.modelType}
                {currentRunMeta.modelVersion ? ` · v${currentRunMeta.modelVersion}` : ''} · job{' '}
                {currentRunMeta.jobId}
            </span>
            <span>
                {currentRunMeta.rows} row{currentRunMeta.rows === 1 ? '' : 's'}
            </span>
            <span>{describeThresholdContext(currentRunMeta.thresholdContext)}</span>
            {currentRunMeta.latencyMs != null && <span>{currentRunMeta.latencyMs} ms</span>}
            {currentRunMeta.status === 'success' && (
                <span>{resultsView === 'table' ? 'table view' : 'list view'}</span>
            )}
            <span className="text-current opacity-70">
                {new Date(currentRunMeta.at).toLocaleString()}
            </span>
        </div>
    )}</>);
}

function PendingRunNotice({ controller }: { controller: Pick<InferenceController, 'activeRun'> }) {
    const { activeRun } = controller;
    return (<>{activeRun && (
        <div
            role="status"
            aria-atomic="true"
            className="mb-3 p-2.5 rounded border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-[11px] flex items-center gap-2"
        >
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
            <span>
                {activeRun.label}
                {activeRun.retryOf ? ' (retry)' : ''} is running…
            </span>
        </div>
    )}</>);
}

function PredictionStatistics({ controller }: { controller: Pick<InferenceController, 'predictionStats'> }) {
    const { predictionStats } = controller;
    return (<>{predictionStats && (
        <div className="mb-3 flex flex-wrap items-center gap-3">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className="inline-flex items-center gap-1 text-gray-400">
                    <BarChart3 className="w-3 h-3" /> Stats
                </span>
                <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200">
                    n = {predictionStats.count}
                </span>
                <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                    mean {predictionStats.mean.toFixed(4)}
                </span>
                <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                    min {predictionStats.min.toFixed(4)}
                </span>
                <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 tabular-nums">
                    max {predictionStats.max.toFixed(4)}
                </span>
            </div>
            {predictionStats.values.length > 1 && (
                <PredictionHistogram values={predictionStats.values} />
            )}
        </div>
    )}</>);
}

function AppliedThresholds({ controller }: { controller: Pick<InferenceController, 'thresholdsApplied'> }) {
    const { thresholdsApplied } = controller;
    return (<>{thresholdsApplied && Object.keys(thresholdsApplied).length > 0 && (
        <div className="mb-3 p-3 rounded bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700">
            <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-2 flex items-center gap-1">
                <Sparkles className="w-3 h-3" /> Thresholds applied
            </div>
            <div className="flex flex-wrap gap-1.5">
                {Object.entries(thresholdsApplied).map(([cls, thr]) => (
                    <span
                        key={cls}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 text-[11px] font-mono"
                    >
                        {cls}: {thr}
                    </span>
                ))}
            </div>
        </div>
    )}</>);
}

function ResultsBody({ controller }: {
    controller: Pick<InferenceController,
        | 'error' | 'predictions' | 'resultsView' | 'parsedInputRows' | 'schemaChips'
        | 'currentRunMeta' | 'lastAttemptRef' | 'handleRetryRun' | 'activeRun'
    >
}) {
    const { error, predictions, resultsView, parsedInputRows, schemaChips } = controller;
    return (<div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        {error ? (<RunErrorState controller={controller} />) : predictions ? (
            resultsView === 'table' && parsedInputRows.length > 0 ? (
                <InputOutputTable
                    rows={parsedInputRows}
                    predictions={predictions}
                />
            ) : (
                <div className="p-4 space-y-2 overflow-auto h-full">
                    {predictions.map((pred, i) => {
                        const probs = asProbabilityMap(pred);
                        return (
                            <div
                                key={i}
                                className="flex items-start gap-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-100 dark:border-gray-700"
                            >
                                <span className="text-xs text-gray-500 w-8 shrink-0 mt-0.5">
                                    #{i + 1}
                                </span>
                                {probs ? (
                                    <div className="flex-1 min-w-0">
                                        <ProbabilityBars probs={probs} />
                                    </div>
                                ) : (
                                    <span className="font-mono font-medium text-blue-600 dark:text-blue-400 break-all">
                                        {renderPrediction(pred)}
                                    </span>
                                )}
                            </div>
                        );
                    })}
                </div>
            )
        ) : (
            <EmptyState
                icon={<Play className="w-12 h-12 opacity-40" />}
                title="Run a prediction to see results here."
                {...(schemaChips.length > 0
                    ? { description: 'Tip: click Sample to load fresh rows from the training dataset.' }
                    : {})}
            />
        )}
    </div>);
}

function RecentRuns({ controller }: { controller: Pick<InferenceController, 'runHistory' | 'recentLatencies' | 'handleClearRunHistory' | 'handleRestoreRun'> }) {
    const { runHistory, recentLatencies, handleClearRunHistory, handleRestoreRun } = controller;
    return (<>{runHistory.length > 0 && (
        <div className="mt-3 border-t border-gray-100 dark:border-gray-700 pt-3">
            <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-400">
                    <History className="w-3 h-3" /> Recent runs
                </div>
                <div className="flex items-center gap-2">
                    <LatencySparkline values={recentLatencies} />
                    <button
                        type="button"
                        onClick={handleClearRunHistory}
                        className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30"
                        title="Clear stored run history from this browser"
                    >
                        <Trash2 className="w-3 h-3" /> Clear history
                    </button>
                </div>
            </div>
            <p className="text-[10px] text-gray-400 mb-2">
                Stored on this device only for {Math.round(HISTORY_TTL_MS / (60 * 60 * 1000))}h,
                kept to the {MAX_RECENT_RUNS} most recent runs — inputs and results, not a server
                record.
            </p>
            <div className="flex flex-wrap gap-2">
                {runHistory.map(run => (
                    <button
                        key={run.runId}
                        onClick={() => handleRestoreRun(run)}
                        className="flex items-center gap-2 text-[11px] px-2 py-1 rounded border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors"
                        title={`Restore input and outcome from ${run.label}`}
                    >
                        {run.status === 'success' ? (
                            <CheckCircle className="w-3 h-3 text-emerald-500 shrink-0" />
                        ) : run.status === 'cancelled' ? (
                            <XCircle className="w-3 h-3 text-gray-400 shrink-0" />
                        ) : (
                            <AlertCircle className="w-3 h-3 text-rose-500 shrink-0" />
                        )}
                        <span className="font-mono">{run.label}</span>
                        <span className="text-gray-400">·</span>
                        <span className="font-mono">{formatTime(run.at)}</span>
                        <span className="text-gray-400">·</span>
                        <span>
                            {run.rows} row{run.rows === 1 ? '' : 's'}
                        </span>
                        {run.latencyMs != null && (
                            <>
                                <span className="text-gray-400">·</span>
                                <span className="tabular-nums">{run.latencyMs} ms</span>
                            </>
                        )}
                    </button>
                ))}
            </div>
        </div>
    )}</>);
}

function runProvenanceClass(status: 'success' | 'cancelled' | 'failure') {
    return status === 'success'
        ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300'
        : status === 'cancelled'
            ? 'bg-gray-50 dark:bg-gray-800/60 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300'
            : 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-800 text-rose-700 dark:text-rose-300';
}

function RunErrorState({ controller }: { controller: Pick<InferenceController, 'error' | 'currentRunMeta' | 'lastAttemptRef' | 'handleRetryRun' | 'activeRun'> }) {
    const { error, currentRunMeta, lastAttemptRef, handleRetryRun, activeRun } = controller;
    if (!error) return null;
    return (
        currentRunMeta?.status === 'cancelled' ? (
            <div
                role="status"
                aria-atomic="true"
                className="p-4 flex flex-col items-center justify-center gap-3 text-center h-full"
            >
                <XCircle className="w-8 h-8 text-gray-400" aria-hidden="true" />
                <p className="text-sm text-gray-600 dark:text-gray-300">{error}</p>
                {lastAttemptRef.current && (
                    <button
                        type="button"
                        onClick={() => void handleRetryRun()}
                        disabled={Boolean(activeRun)}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-700 rounded-md hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <RotateCcw className="w-4 h-4" /> Retry same input
                    </button>
                )}
            </div>
        ) : (
            <div className="h-full flex flex-col">
                <ErrorState error={error} onRetry={lastAttemptRef.current ? handleRetryRun : undefined} />
                <p className="px-4 pb-4 -mt-6 text-xs text-gray-500 dark:text-gray-400 text-center">
                    Your input above hasn&apos;t changed — fix it, or retry the exact same
                    request that failed.
                </p>
            </div>
        )
    );
}
