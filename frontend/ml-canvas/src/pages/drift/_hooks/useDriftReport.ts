import { useCallback, useMemo, useState } from 'react';
import {
    monitoringApi,
    DriftReport,
    DriftThresholds,
    DriftJobOption,
    type DriftMetric,
    type ColumnDrift,
} from '../../../core/api/monitoring';

interface CalculateArgs {
    selectedJob: string;
    file: File;
    job: DriftJobOption | undefined;
    thresholds: DriftThresholds;
}

/** Resolve only the adjustable metrics; unknown metrics retain the saved verdict. */
function metricThreshold(metric: string, thresholds: DriftThresholds): number | undefined {
    switch (metric) {
        case 'psi':
        case 'psi_categorical': return thresholds.psi;
        case 'ks_statistic': return thresholds.ks;
        case 'wasserstein_distance': return thresholds.wasserstein;
        case 'kl_divergence': return thresholds.kl;
        default: return undefined;
    }
}

/** The diagnostic p-value follows the statistic when one is present. */
function evaluateMetric(metric: DriftMetric, column: ColumnDrift, thresholds: DriftThresholds): DriftMetric {
    let hasDrift = metric.has_drift;
    const threshold = metricThreshold(metric.metric, thresholds);
    if (threshold != null) hasDrift = metric.value > threshold;
    if (metric.metric === 'ks_test_p_value') {
        const statistic = column.metrics.find(item => item.metric === 'ks_statistic');
        if (statistic != null) hasDrift = statistic.value > (thresholds.ks ?? statistic.threshold);
    }
    return { ...metric, has_drift: hasDrift };
}

/**
 * Owns the drift report lifecycle:
 *   - submits an upload + job pair to the backend
 *   - re-evaluates `has_drift` per metric on the client when the user nudges
 *     the threshold sliders, so the table updates without a server round trip
 *   - classifies a failed request into `errorKind` so the page can render the
 *     no-baseline (404) and evaluation-failed (5xx) cases explicitly instead
 *     of a single generic error banner (OPS-003)
 */
export function useDriftReport(thresholds: DriftThresholds) {
    const [report, setReport] = useState<DriftReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [errorKind, setErrorKind] = useState<'no_baseline' | 'failed' | null>(null);

    const calculate = useCallback(async ({ selectedJob, file, job, thresholds: t }: CalculateArgs) => {
        setLoading(true);
        setError(null);
        setErrorKind(null);
        try {
            const result = await monitoringApi.calculateDrift(selectedJob, file, job?.dataset_name, t);
            setReport(result);
            return result;
        } catch (err: unknown) {
            const response =
                err && typeof err === 'object' && 'response' in err
                    ? (err as { response?: { status?: number; data?: { detail?: string } } }).response
                    : undefined;
            const detail = response?.data?.detail;
            setError(detail || 'Failed to calculate drift.');
            setErrorKind(response?.status === 404 ? 'no_baseline' : 'failed');
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    /**
     * Re-applies the user's current thresholds to the cached report so the
     * "drifted" flag matches the latest sliders. Pure transform — no fetch.
     */
    const evaluatedReport = useMemo<DriftReport | null>(() => {
        if (!report) return null;
        const t = thresholds;
        const newDrifts: DriftReport['column_drifts'] = {};
        let driftedCount = 0;
        for (const [colName, col] of Object.entries(report.column_drifts)) {
            const newMetrics = col.metrics.map(m => evaluateMetric(m, col, t));
            const drifted = newMetrics.some(m => m.has_drift);
            if (drifted) driftedCount++;
            newDrifts[colName] = { ...col, metrics: newMetrics, drift_detected: drifted };
        }
        // Schema drift counts too (OC-45): the backend includes missing/new
        // columns in `drifted_columns_count`, and rebuilding the count from
        // metric flags alone would silently drop them the moment a slider moved.
        const schemaDriftCount = report.missing_columns.length + report.new_columns.length;
        return {
            ...report,
            column_drifts: newDrifts,
            drifted_columns_count: driftedCount + schemaDriftCount,
        };
    }, [report, thresholds]);

    return {
        report,
        evaluatedReport,
        loading,
        error,
        errorKind,
        setError,
        calculate,
    };
}
