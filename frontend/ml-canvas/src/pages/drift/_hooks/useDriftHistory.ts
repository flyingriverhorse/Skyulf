import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { monitoringApi, DriftHistoryEntry } from '../../../core/api/monitoring';

/**
 * Drift history time-series for a given job, plus a derived per-column PSI
 * series keyed by column name (used by the table sparklines).
 */
export function useDriftHistory(jobId: string) {
    const [driftHistory, setDriftHistory] = useState<DriftHistoryEntry[]>([]);
    const requestId = useRef(0);

    const refresh = useCallback(() => {
        const currentRequest = ++requestId.current;
        if (!jobId) {
            setDriftHistory([]);
            return;
        }
        monitoringApi
            .getDriftHistory(jobId)
            .then(history => { if (currentRequest === requestId.current) setDriftHistory(history); })
            .catch(() => { if (currentRequest === requestId.current) setDriftHistory([]); });
    }, [jobId]);

    useEffect(() => {
        setDriftHistory([]);
        refresh();
        return () => { requestId.current += 1; };
    }, [refresh]);

    /** Per-column PSI series (oldest → newest) for inline sparklines. */
    const columnSparklines = useMemo<Record<string, (number | null)[]>>(() => {
        if (driftHistory.length < 2) return {};
        const reversed = [...driftHistory].reverse();
        const columns = new Set(reversed.flatMap(entry => Object.keys(entry.summary ?? {})));
        return Object.fromEntries([...columns].map(column => [column, reversed.map(entry => {
            const psi = entry.summary?.[column]?.psi;
            return typeof psi === 'number' && Number.isFinite(psi) ? psi : null;
        })]));
    }, [driftHistory]);

    return { driftHistory, columnSparklines, refreshHistory: refresh };
}
