import { useCallback, useEffect, useRef, useState } from 'react';
import { deploymentApi, DeploymentInfo } from '../../../core/api/deployment';
import { SavedThresholdInfo } from '../../../core/api/thresholdTuning';
import {
    isAbortError, loadRunHistory, LS_PENDING_RUN, MAX_RECENT_RUNS, PendingRun,
    persistRunHistory, PREDICT_TIMEOUT_MS, RunOutcome, RunRecord, RunThresholdContext,
} from './inferenceData';

/** Own named prediction requests, cancellation, retry, and durable run history. */
export function useInferenceRuns(activeDeployment: DeploymentInfo | null, savedThresholds: SavedThresholdInfo | null, setInputData: (input: string) => void) {
    /** Atomic guard against a second submission racing the pending-state
     * update — checked synchronously before any async work starts. */
    const activeRunIdRef = useRef<string | null>(null);
    const activeAbortControllerRef = useRef<AbortController | null>(null);
    const activeTimeoutRef = useRef<number | null>(null);
    /** Distinguishes a user-initiated Cancel from our own timeout-abort,
     * since both surface to the catch block as the same cancellation error. */
    const cancelReasonRef = useRef<'user' | 'timeout' | null>(null);
    const runSeqRef = useRef(0);
    /** Exact payload of the most recent submission, so "Retry" resends the
     * request that actually failed rather than whatever is currently typed. */
    const lastAttemptRef = useRef<{
        data: unknown[];
        overrideThresholds: Record<string, number> | null;
        inputSnapshot: string;
        runId: string;
    } | null>(null);
    const [predictions, setPredictions] = useState<unknown[] | null>(null);
    /** The one run currently in flight, if any — naming it prevents a second
     * submission from starting while this one is still pending. */
    const [activeRun, setActiveRun] = useState<PendingRun | null>(null);
    /** Provenance for whichever settled run is currently shown in the results
     * pane (success, failure, or cancellation) — survives across reload via
     * `runHistory` hydration below. */
    const [currentRunMeta, setCurrentRunMeta] = useState<RunRecord | null>(null);
    const [runHistory, setRunHistory] = useState<RunRecord[]>(() => loadRunHistory());
    const [error, setError] = useState<string | null>(null);
    const [latencyMs, setLatencyMs] = useState<number | null>(null);
    const [thresholdsApplied, setThresholdsApplied] = useState<Record<string, number> | null>(null);

    /** Retire a request before aborting so late transport callbacks cannot write. */
    const invalidateRun = useCallback(() => {
        activeRunIdRef.current = null;
        activeAbortControllerRef.current?.abort();
        activeAbortControllerRef.current = null;
        cancelReasonRef.current = null;
        if (activeTimeoutRef.current !== null) window.clearTimeout(activeTimeoutRef.current);
        activeTimeoutRef.current = null;
    }, []);

    useEffect(() => {
        setActiveRun(null);
        return invalidateRun;
    }, [activeDeployment?.job_id, invalidateRun]);

    /** Clear both the displayed result and any request that could restore it. */
    const clearResults = useCallback(() => {
        invalidateRun();
        clearPendingRun();
        lastAttemptRef.current = null;
        setActiveRun(null);
        setPredictions(null);
        setCurrentRunMeta(null);
        setError(null);
        setLatencyMs(null);
        setThresholdsApplied(null);
    }, [invalidateRun]);

    /** Append a settled run to the durable, reload-surviving history. */
    const appendRunHistory = useCallback((entry: RunRecord) => {
        setRunHistory(prev => {
            const next = [entry, ...prev].slice(0, MAX_RECENT_RUNS);
            persistRunHistory(next);
            return next;
        });
    }, []);

    // On mount: if a run was in flight when the page was last closed/reloaded,
    // its outcome is genuinely unknown — surface that explicitly rather than
    // silently forgetting it (the localStorage marker itself is never a raw
    // transport object, just the same provenance any settled run gets).
    useEffect(() => {
        let interrupted:
            | (Omit<RunRecord, 'status' | 'errorMessage' | 'predictions' | 'latencyMs'> & { at: number })
            | null = null;
        try {
            const raw = localStorage.getItem(LS_PENDING_RUN);
            if (raw) interrupted = JSON.parse(raw);
        } catch {
            interrupted = null;
        }
        try {
            localStorage.removeItem(LS_PENDING_RUN);
        } catch {
            /* ignore */
        }
        if (interrupted) {
            const entry: RunRecord = {
                ...interrupted,
                status: 'failure',
                latencyMs: null,
                predictions: null,
                errorMessage:
                    'This run was still in progress when the page was reloaded or closed — its outcome is unknown. Retry to get a fresh result.',
            };
            setCurrentRunMeta(entry);
            setError(entry.errorMessage);
            appendRunHistory(entry);
            return;
        }
        // Otherwise hydrate the results pane from the most recent surviving
        // history entry so a reload doesn't wipe evidence the user still needs.
        const latest = loadRunHistory()[0];
        if (latest) {
            setCurrentRunMeta(latest);
            if (latest.status === 'success') {
                setPredictions(latest.predictions);
                setLatencyMs(latest.latencyMs);
                setThresholdsApplied(latest.overrideThresholdsUsed);
            } else {
                setError(latest.errorMessage);
            }
        }
        // Mount-only hydration — intentionally excludes appendRunHistory from
        // deps since it's stable and re-running this on every render would
        // re-import the interrupted-run marker.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    /**
     * Core run executor shared by "Run Prediction" and "Retry" — always
     * named, cancellable, and timed-out, and always settles into exactly one
     * durable `RunRecord` (never a bare error string or raw transport object).
     */
    const submitRun = useCallback(
        async (
            data: unknown[],
            overrideThresholds: Record<string, number> | null,
            inputSnapshot: string,
            opts?: { retryOf?: string },
        ) => {
            if (!activeDeployment || activeRunIdRef.current) return;

            const seq = ++runSeqRef.current;
            const runId = `run-${Date.now().toString(36)}-${seq}`;
            const label = `Run #${seq}`;
            const thresholdContext = getRunThresholdContext(overrideThresholds, savedThresholds);

            const controller = new AbortController();
            activeAbortControllerRef.current = controller;
            activeRunIdRef.current = runId;
            cancelReasonRef.current = null;
            lastAttemptRef.current = { data, overrideThresholds, inputSnapshot, runId };

            const pendingMarker = {
                runId,
                label,
                at: Date.now(),
                rows: data.length,
                jobId: activeDeployment.job_id,
                modelType: activeDeployment.model_type,
                modelVersion: null,
                thresholdContext,
                input: inputSnapshot,
                overrideThresholdsUsed: overrideThresholds,
            };
            persistPendingRun(pendingMarker);

            const timeoutId = window.setTimeout(() => {
                cancelReasonRef.current = 'timeout';
                controller.abort();
            }, PREDICT_TIMEOUT_MS);
            activeTimeoutRef.current = timeoutId;

            setActiveRun({ runId, label, submittedAt: Date.now(), retryOf: opts?.retryOf ?? null });
            setError(null);

            const start = performance.now();
            try {
                const response = await deploymentApi.predict(data, overrideThresholds, {
                    signal: controller.signal,
                });
                if (activeRunIdRef.current !== runId) return;
                throwIfRunCancelled(controller.signal);
                const elapsed = Math.round(performance.now() - start);
                setPredictions(response.predictions);
                setLatencyMs(elapsed);
                setThresholdsApplied(response.thresholds_applied ?? null);
                const entry: RunRecord = {
                    runId,
                    label,
                    status: 'success',
                    at: Date.now(),
                    rows: data.length,
                    latencyMs: elapsed,
                    jobId: activeDeployment.job_id,
                    modelType: activeDeployment.model_type,
                    modelVersion: response.model_version ?? null,
                    thresholdContext,
                    input: inputSnapshot,
                    overrideThresholdsUsed: overrideThresholds,
                    predictions: response.predictions,
                    errorMessage: null,
                };
                setCurrentRunMeta(entry);
                appendRunHistory(entry);
            } catch (e: unknown) {
                if (activeRunIdRef.current !== runId) return;
                const { status, message } = describeRunFailure(e, cancelReasonRef.current);
                setPredictions(null);
                setLatencyMs(null);
                setThresholdsApplied(null);
                setError(message);
                const entry: RunRecord = {
                    runId,
                    label,
                    status,
                    at: Date.now(),
                    rows: data.length,
                    latencyMs: null,
                    jobId: activeDeployment.job_id,
                    modelType: activeDeployment.model_type,
                    modelVersion: null,
                    thresholdContext,
                    input: inputSnapshot,
                    overrideThresholdsUsed: overrideThresholds,
                    predictions: null,
                    errorMessage: message,
                };
                setCurrentRunMeta(entry);
                appendRunHistory(entry);
            } finally {
                window.clearTimeout(timeoutId);
                if (activeRunIdRef.current === runId) {
                    cancelReasonRef.current = null;
                    activeAbortControllerRef.current = null;
                    activeRunIdRef.current = null;
                    activeTimeoutRef.current = null;
                    setActiveRun(null);
                    clearPendingRun();
                }
            }
        },
        [activeDeployment, savedThresholds, appendRunHistory],
    );

    /** Re-run the exact request that failed/was cancelled — same input, same
     * override thresholds — as a new named run. */
    const handleRetryRun = useCallback(async () => {
        const attempt = lastAttemptRef.current;
        if (!attempt || activeRun) return;
        await submitRun(attempt.data, attempt.overrideThresholds, attempt.inputSnapshot, {
            retryOf: attempt.runId,
        });
    }, [activeRun, submitRun]);

    /** Abort the in-flight run — distinguished from our own timeout-abort via `cancelReasonRef`. */
    const handleCancelRun = useCallback(() => {
        if (!activeAbortControllerRef.current) return;
        cancelReasonRef.current = 'user';
        activeAbortControllerRef.current.abort();
    }, []);

    const handleRestoreRun = (run: RunRecord) => {
        setInputData(run.input);
        setCurrentRunMeta(run);
        if (run.status === 'success') {
            setPredictions(run.predictions);
            setLatencyMs(run.latencyMs);
            setThresholdsApplied(run.overrideThresholdsUsed);
            setError(null);
        } else {
            setPredictions(null);
            setLatencyMs(null);
            setThresholdsApplied(null);
            setError(run.errorMessage);
        }
    };

    /** Remove the durable run history (both in-memory and localStorage) —
     * does not touch the currently displayed run. */
    const handleClearRunHistory = () => {
        setRunHistory([]);
        persistRunHistory([]);
    };
    return {
        lastAttemptRef, predictions, setPredictions, activeRun, currentRunMeta,
        setCurrentRunMeta, runHistory, setRunHistory, error, setError, latencyMs,
        setLatencyMs, thresholdsApplied, setThresholdsApplied, submitRun, handleRetryRun,
        handleCancelRun, handleRestoreRun, handleClearRunHistory, clearResults,
    };
}

/** Remove a pending marker only while its request still owns the result state. */
function clearPendingRun() {
    try {
        localStorage.removeItem(LS_PENDING_RUN);
    } catch {
        /* ignore */
    }
}

/** Match transport cancellation when a response was already queued before abort. */
function throwIfRunCancelled(signal: AbortSignal) {
    if (signal.aborted) throw Object.assign(new Error('Run cancelled'), { code: 'ERR_CANCELED' });
}

/** Capture the threshold source in effect when a run starts. */
function getRunThresholdContext(
    overrideThresholds: Record<string, number> | null,
    savedThresholds: SavedThresholdInfo | null,
): RunThresholdContext {
    if (overrideThresholds) return 'override';
    return savedThresholds?.enabled ? 'saved-enabled' : 'none';
}

/** Distinguish cancellation, timeout, and server failures for run provenance. */
function describeRunFailure(e: unknown, cancelReason: 'user' | 'timeout' | null) {
    const canceled = isAbortError(e);
    const timedOut = canceled && cancelReason === 'timeout';
    const status: RunOutcome = canceled && !timedOut ? 'cancelled' : 'failure';
    const message = timedOut
        ? `The server did not respond within ${Math.round(PREDICT_TIMEOUT_MS / 1000)}s. Your input is unchanged — retry, or check the deployment.`
        : status === 'cancelled'
            ? 'Run cancelled — your input is unchanged.'
            : (e as Error).message || 'Prediction failed';
    return { status, message };
}

/** Keep enough request provenance to explain a run interrupted by reload. */
function persistPendingRun(pendingMarker: Omit<RunRecord, 'status' | 'latencyMs' | 'predictions' | 'errorMessage'>) {
    try {
        localStorage.setItem(LS_PENDING_RUN, JSON.stringify(pendingMarker));
    } catch {
        /* ignore */
    }
}
