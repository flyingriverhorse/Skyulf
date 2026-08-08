import { useCallback, useEffect, useState } from 'react';
import {
    monitoringApi,
    DriftAlertDetail,
    DriftDispositionAction,
} from '../../../core/api/monitoring';

/**
 * Owns a single drift alert's full detail (evidence + disposition history)
 * and the acknowledge/resolve/reopen action that mutates it. `alertId` of
 * `null`/`undefined` clears the detail rather than fetching — used while no
 * row is being investigated.
 */
export function useDriftAlertDetail(alertId: number | null | undefined) {
    const [detail, setDetail] = useState<DriftAlertDetail | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [actionPending, setActionPending] = useState(false);

    const refresh = useCallback(async () => {
        if (alertId == null) {
            setDetail(null);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const result = await monitoringApi.getDriftAlert(alertId);
            setDetail(result);
        } catch {
            setError('Failed to load drift alert detail.');
        } finally {
            setLoading(false);
        }
    }, [alertId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const applyDisposition = useCallback(
        async (action: DriftDispositionAction, actor: string, note?: string) => {
            if (alertId == null) return null;
            setActionPending(true);
            setError(null);
            try {
                const result = await monitoringApi.updateDriftAlertDisposition(
                    alertId,
                    action,
                    actor,
                    note,
                );
                setDetail(result);
                return result;
            } catch (err: unknown) {
                const responseDetail =
                    err && typeof err === 'object' && 'response' in err
                        ? (err as { response?: { data?: { detail?: string } } }).response?.data
                              ?.detail
                        : undefined;
                setError(responseDetail || `Failed to ${action} the drift alert.`);
                return null;
            } finally {
                setActionPending(false);
            }
        },
        [alertId],
    );

    return { detail, loading, error, actionPending, refresh, applyDisposition };
}
