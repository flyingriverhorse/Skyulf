import { useCallback, useMemo, useState } from 'react';
import {
    monitoringApi,
    DriftReport,
    DriftThresholds,
    DriftJobOption,
} from '../../../core/api/monitoring';

interface CalculateArgs {
    selectedJob: string;
    file: File;
    job: DriftJobOption | undefined;
    thresholds: DriftThresholds;
}

/**
 * Owns the drift report lifecycle:
 *   - submits an upload + job pair to the backend
 *   - re-evaluates `has_drift` per metric on the client when the user nudges
 *     the threshold sliders, so the table updates without a server round trip
 */
export function useDriftReport(thresholds: DriftThresholds) {
    const [report, setReport] = useState<DriftReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const calculate = useCallback(async ({ selectedJob, file, job, thresholds: t }: CalculateArgs) => {
        setLoading(true);
        setError(null);
        try {
            const result = await monitoringApi.calculateDrift(selectedJob, file, job?.dataset_name, t);
            setReport(result);
            return result;
        } catch (err: unknown) {
            const detail =
                err && typeof err === 'object' && 'response' in err
                    ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
                    : undefined;
            setError(detail || 'Failed to calculate drift.');
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
            const newMetrics = col.metrics.map(m => {
                let hasDrift = m.has_drift;
                if (m.metric === 'psi' && t.psi != null) hasDrift = m.value > t.psi;
                if (m.metric === 'ks_test_p_value' && t.ks != null) hasDrift = m.value < t.ks;
                if (m.metric === 'wasserstein_distance' && t.wasserstein != null)
                    hasDrift = m.value > t.wasserstein;
                if (m.metric === 'kl_divergence' && t.kl != null) hasDrift = m.value > t.kl;
                return { ...m, has_drift: hasDrift };
            });
            const drifted = newMetrics.some(m => m.has_drift);
            if (drifted) driftedCount++;
            newDrifts[colName] = { ...col, metrics: newMetrics, drift_detected: drifted };
        }
        return { ...report, column_drifts: newDrifts, drifted_columns_count: driftedCount };
    }, [report, thresholds]);

    return {
        report,
        evaluatedReport,
        loading,
        error,
        setError,
        calculate,
    };
}
