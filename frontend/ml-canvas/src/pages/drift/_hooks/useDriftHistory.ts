import { useCallback, useEffect, useMemo, useState } from 'react';
import { monitoringApi, DriftHistoryEntry } from '../../../core/api/monitoring';

/**
 * Drift history time-series for a given job, plus a derived per-column PSI
 * series keyed by column name (used by the table sparklines).
 */
export function useDriftHistory(jobId: string) {
    const [driftHistory, setDriftHistory] = useState<DriftHistoryEntry[]>([]);

    const refresh = useCallback(() => {
        if (!jobId) {
            setDriftHistory([]);
            return;
        }
        monitoringApi
            .getDriftHistory(jobId)
            .then(setDriftHistory)
            .catch(() => setDriftHistory([]));
    }, [jobId]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    /** Per-column PSI series (oldest → newest) for inline sparklines. */
    const columnSparklines = useMemo<Record<string, number[]>>(() => {
        if (driftHistory.length < 2) return {};
        const reversed = [...driftHistory].reverse();
        const result: Record<string, number[]> = {};
        for (const entry of reversed) {
            if (!entry.summary) continue;
            for (const [col, data] of Object.entries(entry.summary)) {
                if (!result[col]) result[col] = [];
                result[col].push(data.psi ?? 0);
            }
        }
        return result;
    }, [driftHistory]);

    return { driftHistory, columnSparklines, refreshHistory: refresh };
}
