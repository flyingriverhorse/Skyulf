import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDeploymentInput } from './useDeploymentInput';
import { useInferenceRuns } from './useInferenceRuns';

import { deploymentApi } from '../../../core/api/deployment';
import { toast } from '../../../core/toast';
import { useConfirm } from '../../shared';

import {
    analyseInput, asProbabilityMap, checkSchema, DEFAULT_INPUT, LARGE_BATCH_THRESHOLD,
    LS_INPUT, LS_VIEW, parseCsv, persistRunHistory, renderPrediction, rowsToCsv,
    SchemaCheck, toNumericArray,
} from './inferenceData';
import { useSavedThresholdInfo } from './useSavedThresholdInfo';

/** Connect inference input, deployment, run state, and existing screen actions. */
export function useInferenceController() {
    const confirm = useConfirm();
    const csvInputRef = useRef<HTMLInputElement>(null);
    const editorWrapRef = useRef<HTMLDivElement>(null);
    const [inputData, setInputData] = useState<string>(() => {
        try {
            return localStorage.getItem(LS_INPUT) ?? DEFAULT_INPUT;
        } catch {
            return DEFAULT_INPUT;
        }
    });
    const {
        activeDeployment, setActiveDeployment, datasetId, setDatasetId, excludedColumns,
        setExcludedColumns, isReloadingSample, sampleSize, setSampleSize, autoFilterInfo,
        schemaChips, handleReloadSample,
    } = useDeploymentInput(setInputData);

    /** Ad-hoc per-class decision threshold overrides for this prediction only. */
    const [overrideThresholdsEnabled, setOverrideThresholdsEnabled] = useState(false);
    const [overrideThresholdsValue, setOverrideThresholdsValue] = useState<Record<string, number>>({});
    const [newOverrideClass, setNewOverrideClass] = useState('');
    const [newOverrideThreshold, setNewOverrideThreshold] = useState('0.5');

    /** Tuned thresholds already saved for the active deployment's job (from the
     * Evaluation tab's Threshold Tuning panel) — these are applied automatically
     * at /predict time whenever `enabled` is true and no ad-hoc override is set. */
    const savedThresholds = useSavedThresholdInfo(activeDeployment?.job_id ?? null);
    const {
        lastAttemptRef, predictions, setPredictions, activeRun, currentRunMeta,
        setCurrentRunMeta, runHistory, setRunHistory, error, setError, latencyMs,
        setLatencyMs, thresholdsApplied, setThresholdsApplied, submitRun, handleRetryRun,
        handleCancelRun, handleRestoreRun, handleClearRunHistory,
    } = useInferenceRuns(activeDeployment, savedThresholds, setInputData);
    const [bannerDismissed, setBannerDismissed] = useState(false);

    const [resultsView, setResultsView] = useState<'list' | 'table'>(() => {
        try {
            const v = localStorage.getItem(LS_VIEW);
            return v === 'table' ? 'table' : 'list';
        } catch {
            return 'list';
        }
    });
    const [isDragging, setIsDragging] = useState(false);

    // EXP-006: missing schema fields block Run Prediction by default — the
    // backend rejects (or, when a field is also wrong-typed, crashes on)
    // exactly this request shape, so silently letting it through just moves
    // the failure one network round-trip later. An explicit, reviewable
    // override is offered instead of a hard block, since a user may
    // legitimately want to send a partial row (e.g. relying on a
    // server-side default) and know what they're doing.
    const [acknowledgeMissingFields, setAcknowledgeMissingFields] = useState(false);

    const inputStatus = useMemo(() => analyseInput(inputData), [inputData]);

    const schemaCheck = useMemo(
        () => checkSchema(inputData, schemaChips),
        [inputData, schemaChips],
    );

    // Any field the deployment schema can't type (the common case today —
    // the artifact-derived schema currently reports every field as
    // `unknown`) never gets a value/type check below; be explicit about
    // that instead of silently skipping it, per EXP-006's "unknown schema
    // types are honestly marked unvalidated" requirement.
    const hasUnknownTypedFields = useMemo(
        () => schemaChips.some(col => col.type === 'unknown'),
        [schemaChips],
    );

    // Re-require acknowledgement whenever the actual set of missing rows/
    // fields changes (edit, Fix, new sample, etc.) — a stale checkbox
    // ticked for a previous violation must not silently authorize a new one.
    const missingSignature = schemaCheck?.rowIssues
        .map(i => `${i.rowIndex}:${i.missing.join(',')}`)
        .join('|') ?? '';
    useEffect(() => {
        setAcknowledgeMissingFields(false);
    }, [missingSignature]);

    const predictionStats = useMemo(() => {
        if (!predictions || predictions.length === 0) return null;
        const nums = toNumericArray(predictions);
        if (nums.length === 0) return null;
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
        return { count: nums.length, mean, min, max, values: nums };
    }, [predictions]);

    const parsedInputRows = useMemo(() => {
        if (!inputStatus.valid) return [];
        try {
            const arr = JSON.parse(inputData);
            return Array.isArray(arr) ? (arr as unknown[]) : [];
        } catch {
            return [];
        }
    }, [inputStatus.valid, inputData]);

    /** Detect "this is a single classification probability response" shape. */
    const singleProbMap = useMemo(() => {
        if (!predictions || predictions.length !== 1) return null;
        return asProbabilityMap(predictions[0]);
    }, [predictions]);

    /** Persist user preferences across reloads. */
    useEffect(() => {
        try {
            localStorage.setItem(LS_INPUT, inputData);
        } catch {
            /* storage may be disabled in private mode — ignore */
        }
    }, [inputData]);
    useEffect(() => {
        try {
            localStorage.setItem(LS_VIEW, resultsView);
        } catch {
            /* ignore */
        }
    }, [resultsView]);

    /** Shared CSV-text → JSON-array writer (used by file picker and DnD). */
    const ingestCsvText = useCallback(
        (text: string) => {
            try {
                const parsed = parseCsv(text);
                if (parsed.length === 0) {
                    toast.error('CSV had no data rows');
                    return;
                }
                const rows = parsed.map(r => {
                    if (excludedColumns.size === 0) return r;
                    const cleaned: Record<string, unknown> = {};
                    Object.entries(r).forEach(([k, v]) => {
                        if (!excludedColumns.has(k)) cleaned[k] = v;
                    });
                    return cleaned;
                });
                setInputData(JSON.stringify(rows, null, 2));
                toast.success(`Loaded ${rows.length} row${rows.length === 1 ? '' : 's'} from CSV`);
            } catch (e) {
                console.error('CSV parse failed', e);
                toast.error('Could not parse CSV');
            }
        },
        [excludedColumns],
    );

    const handleCsvFile = useCallback(
        (file: File) => {
            const reader = new FileReader();
            reader.onload = () => ingestCsvText(String(reader.result ?? ''));
            reader.onerror = () => toast.error('Could not read file');
            reader.readAsText(file);
        },
        [ingestCsvText],
    );

    const handleCsvChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleCsvFile(file);
        e.target.value = '';
    };

    /** Drag-and-drop CSV onto the editor. */
    const handleDragOver = (e: React.DragEvent) => {
        if (e.dataTransfer.types.includes('Files')) {
            e.preventDefault();
            setIsDragging(true);
        }
    };
    const handleDragLeave = (e: React.DragEvent) => {
        // Only clear when leaving the wrapper itself, not its children.
        if (e.currentTarget === e.target) setIsDragging(false);
    };
    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.csv')) {
            toast.error('Only .csv files are supported');
            return;
        }
        handleCsvFile(file);
    };

    /** Pad missing schema fields with zero in every row of the input — after
     * an explicit reviewed preview of exactly which row/field pairs will be
     * touched, since the previous silent apply made a partial repair look
     * like a full fix (a field that already existed with a wrong-typed
     * value was left untouched and only surfaced once the backend crashed
     * on it — see EXP-006). */
    const handleFixMissingFields = async () => {
        if (!schemaCheck || schemaCheck.rowIssues.length === 0) return;
        const preview = schemaCheck.rowIssues
            .slice(0, 8)
            .map(issue => `Row ${issue.rowIndex + 1}: ${issue.missing.join(', ')} → 0`);
        const more = schemaCheck.rowIssues.length > 8 ? schemaCheck.rowIssues.length - 8 : 0;
        const ok = await confirm({
            title: 'Fill missing fields with 0?',
            message: (
                <div className="space-y-2">
                    <p>
                        This only fills in fields that are absent from a row. Fields that are
                        already present keep their current value even if it looks wrong for
                        this model — review those yourself before running.
                    </p>
                    <ul className="text-xs font-mono bg-gray-50 dark:bg-gray-900 rounded p-2 space-y-0.5 max-h-32 overflow-auto">
                        {preview.map(line => (
                            <li key={line}>{line}</li>
                        ))}
                        {more > 0 && <li>…and {more} more row(s)</li>}
                    </ul>
                </div>
            ),
            confirmLabel: 'Fill with 0',
        });
        if (!ok) return;
        try {
            const arr = JSON.parse(inputData) as Record<string, unknown>[];
            const padded = arr.map(row => {
                const next = { ...row };
                schemaCheck.missing.forEach(field => {
                    if (!(field in next)) next[field] = 0;
                });
                return next;
            });
            setInputData(JSON.stringify(padded, null, 2));
            toast.success(`Filled ${schemaCheck.missing.length} missing field(s) with 0 — verify values before running`);
        } catch {
            toast.error('Cannot pad: invalid JSON');
        }
    };

    const handleDeactivate = async () => {
        const ok = await confirm({
            title: 'Undeploy model?',
            message: 'Are you sure you want to undeploy the current model?',
            confirmLabel: 'Undeploy',
            variant: 'danger',
        });
        if (!ok) return;
        try {
            await deploymentApi.deactivate();
            setActiveDeployment(null);
            setDatasetId(null);
            setExcludedColumns(new Set());
            setPredictions(null);
            setLatencyMs(null);
            setCurrentRunMeta(null);
            setRunHistory([]);
            persistRunHistory([]);
        } catch (e) {
            console.error('Failed to deactivate', e);
            toast.error('Failed to undeploy model');
        }
    };

    const handlePredict = useCallback(async () => {
        if (!activeDeployment || activeRun) return;
        // EXP-006: a row missing a schema field is the same partial request
        // the backend either rejects or crashes on — require the explicit
        // acknowledgement checkbox before sending it, even via Ctrl+Enter
        // (which otherwise bypasses the disabled Run Prediction button).
        if (hasUnacknowledgedFields(schemaCheck, acknowledgeMissingFields)) {
            return;
        }

        // Soft warning before sending huge payloads — the network round-trip
        // and JSON serialisation balloon, and it's almost always a mistake.
        let data: unknown[];
        try {
            data = parsePredictionData(inputData);
        } catch (e) {
            setError((e as Error).message);
            return;
        }
        if (data.length > LARGE_BATCH_THRESHOLD && !await confirmLargeBatch(data.length, confirm)) return;

        await submitRun(data, overrideThresholdsEnabled ? overrideThresholdsValue : null, inputData);
    }, [
        activeDeployment,
        activeRun,
        inputData,
        confirm,
        overrideThresholdsEnabled,
        overrideThresholdsValue,
        schemaCheck,
        acknowledgeMissingFields,
        submitRun,
        setError,
    ]);

    const handleFormatJson = () => {
        try {
            const parsed = JSON.parse(inputData);
            setInputData(JSON.stringify(parsed, null, 2));
        } catch {
            toast.error('Cannot format: invalid JSON');
        }
    };

    /** Wipe input + results back to the empty default. */
    const handleClearInput = () => {
        setInputData(DEFAULT_INPUT);
        setPredictions(null);
        setError(null);
        setLatencyMs(null);
        setThresholdsApplied(null);
        setCurrentRunMeta(null);
    };

    const handleTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            void handlePredict();
        }
    };

    /** Add (or update) a single class → threshold pair in the override editor. */
    const handleAddOverrideEntry = () => {
        const cls = newOverrideClass.trim();
        const threshold = Number(newOverrideThreshold);
        if (!cls || !Number.isFinite(threshold)) return;
        setOverrideThresholdsValue(prev => ({ ...prev, [cls]: threshold }));
        setNewOverrideClass('');
        setNewOverrideThreshold('0.5');
    };

    /** Remove a class → threshold pair from the override editor. */
    const handleRemoveOverrideEntry = (cls: string) => {
        setOverrideThresholdsValue(prev => {
            const next = { ...prev };
            delete next[cls];
            return next;
        });
    };

    /** Update the threshold value for an existing override entry. */
    const handleOverrideThresholdChange = (cls: string, value: string) => {
        const num = Number(value);
        if (!Number.isFinite(num)) return;
        setOverrideThresholdsValue(prev => ({ ...prev, [cls]: num }));
    };

    /** Copy the deployment's already-saved tuned thresholds into the ad-hoc
     * override editor and enable it — lets the user start from what's
     * already active server-side and tweak individual classes instead of
     * typing every value from scratch. */
    const handlePrefillFromSavedThresholds = () => {
        if (!savedThresholds?.thresholds) return;
        setOverrideThresholdsValue({ ...savedThresholds.thresholds });
        setOverrideThresholdsEnabled(true);
    };

    /** Pre-fill the override editor with classes seen in the last single-row
     * prediction's probability map (the only reliable class-list source on
     * this page, since deployments don't expose `estimator.classes_`). */
    const handlePrefillFromLastPrediction = () => {
        if (!singleProbMap) return;
        setOverrideThresholdsValue(prev => {
            const next = { ...prev };
            for (const cls of Object.keys(singleProbMap)) {
                if (!(cls in next)) next[cls] = 0.5;
            }
            return next;
        });
    };

    const handleCopyPredictions = async () => {
        if (!predictions) return;
        try {
            await navigator.clipboard.writeText(JSON.stringify(predictions, null, 2));
            toast.success('Copied predictions to clipboard');
        } catch {
            toast.error('Failed to copy');
        }
    };

    /** Trigger a file download for the given Blob. */
    const triggerDownload = (blob: Blob, filename: string) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    /** Filename stem carrying run provenance (job + run label), so an
     * exported file can be traced back to the run that produced it even
     * once it's out of the browser. */
    const exportFileStem = useMemo(() => {
        const job = currentRunMeta?.jobId ?? 'unknown-job';
        const runId = currentRunMeta?.runId ?? `run-${Date.now()}`;
        return `predictions_${job}_${runId}`;
    }, [currentRunMeta]);

    const handleDownloadJson = () => {
        if (!predictions) return;
        triggerDownload(
            new Blob([JSON.stringify(predictions, null, 2)], { type: 'application/json' }),
            `${exportFileStem}.json`,
        );
    };

    /** Export inputs + predictions side-by-side to CSV. */
    const handleDownloadCsv = () => {
        if (!predictions) return;
        const merged: Record<string, unknown>[] = parsedInputRows.map((row, i) => {
            const obj = (row && typeof row === 'object'
                ? { ...(row as Record<string, unknown>) }
                : {}) as Record<string, unknown>;
            obj.prediction = renderPrediction(predictions[i]);
            return obj;
        });
        // Fallback: predictions-only column when there are no parsed input rows.
        const rows =
            merged.length > 0
                ? merged
                : predictions.map((p, i) => ({ row: i + 1, prediction: renderPrediction(p) }));
        triggerDownload(
            new Blob([rowsToCsv(rows)], { type: 'text/csv;charset=utf-8' }),
            `${exportFileStem}.csv`,
        );
    };

    const showBanner = autoFilterInfo && !bannerDismissed;
    const hasSchemaIssues =
        schemaCheck && (schemaCheck.missing.length > 0 || schemaCheck.extra.length > 0);
    // Missing fields are the one violation we can determine with certainty
    // from a nameless "unknown"-typed schema — block the request on those
    // unless the user has explicitly reviewed and accepted the current set.
    const hasBlockingMissingFields = Boolean(schemaCheck && schemaCheck.rowIssues.length > 0);
    const canRunPrediction =
        !hasBlockingMissingFields || acknowledgeMissingFields;
    const recentLatencies = useMemo(
        () =>
            [...runHistory]
                .reverse()
                .map(r => r.latencyMs)
                .filter((v): v is number => v != null),
        [runHistory],
    );
    return {
        csvInputRef, editorWrapRef, lastAttemptRef, activeDeployment, datasetId,
        excludedColumns, inputData, setInputData, predictions, activeRun, currentRunMeta,
        runHistory, isReloadingSample, sampleSize, setSampleSize, error, latencyMs,
        thresholdsApplied, overrideThresholdsEnabled, setOverrideThresholdsEnabled,
        overrideThresholdsValue, newOverrideClass, setNewOverrideClass,
        newOverrideThreshold, setNewOverrideThreshold, savedThresholds, autoFilterInfo,
        setBannerDismissed, resultsView, setResultsView, isDragging,
        acknowledgeMissingFields, setAcknowledgeMissingFields, inputStatus, schemaChips,
        schemaCheck, hasUnknownTypedFields, predictionStats, parsedInputRows, singleProbMap,
        handleReloadSample, handleCsvChange, handleDragOver, handleDragLeave, handleDrop,
        handleFixMissingFields, handleDeactivate, handlePredict, handleRetryRun,
        handleCancelRun, handleFormatJson, handleClearInput, handleTextareaKeyDown,
        handleAddOverrideEntry, handleRemoveOverrideEntry, handleOverrideThresholdChange,
        handlePrefillFromSavedThresholds, handlePrefillFromLastPrediction,
        handleCopyPredictions, handleDownloadJson, handleDownloadCsv, handleRestoreRun,
        handleClearRunHistory, showBanner, hasSchemaIssues, hasBlockingMissingFields,
        canRunPrediction, recentLatencies,
    };
}

export type InferenceController = ReturnType<typeof useInferenceController>;

/** Apply the same missing-field guard to button and keyboard submission. */
function hasUnacknowledgedFields(schemaCheck: SchemaCheck | null, acknowledged: boolean) {
    return schemaCheck && schemaCheck.rowIssues.length > 0 && !acknowledged;
}

/** Request confirmation only for batches above the size limit. */
async function confirmLargeBatch(rows: number, confirm: ReturnType<typeof useConfirm>) {
    return confirm({
        title: 'Large batch',
        message: `You are about to send ${rows} rows. Continue?`,
        confirmLabel: 'Send',
    });
}

/** Parse a prediction batch while preserving the editor validation message. */
function parsePredictionData(raw: string): unknown[] {
    const data: unknown = JSON.parse(raw);
    if (!Array.isArray(data)) throw new Error('Input must be a JSON array of objects');
    return data;
}
