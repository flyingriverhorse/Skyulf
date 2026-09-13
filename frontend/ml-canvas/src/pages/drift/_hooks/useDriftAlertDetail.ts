import { useCallback, useEffect, useRef, useState } from 'react';
import {
    monitoringApi,
    DriftAlertDetail,
    DriftDispositionAction,
} from '../../../core/api/monitoring';

/** Preserve backend disposition messages while keeping request ownership checks separate. */
function dispositionError(err: unknown, action: DriftDispositionAction): string {
    const responseDetail =
        err && typeof err === 'object' && 'response' in err
            ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
            : undefined;
    return responseDetail || `Failed to ${action} the drift alert.`;
}

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
    const detailRequest = useRef(0);
    const actionRequest = useRef(0);

    const refresh = useCallback(async () => {
        const request = ++detailRequest.current;
        if (alertId == null) {
            setDetail(null);
            setLoading(false);
            setError(null);
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const result = await monitoringApi.getDriftAlert(alertId);
            if (request === detailRequest.current) setDetail(result);
        } catch {
            if (request === detailRequest.current) setError('Failed to load drift alert detail.');
        } finally {
            if (request === detailRequest.current) setLoading(false);
        }
    }, [alertId]);

    useEffect(() => {
        setDetail(null);
        setActionPending(false);
        void refresh();
        return () => {
            detailRequest.current += 1;
            actionRequest.current += 1;
        };
    }, [refresh]);

    const applyDisposition = useCallback(
        async (action: DriftDispositionAction, actor: string, note?: string) => {
            if (alertId == null) return null;
            const request = ++actionRequest.current;
            setActionPending(true);
            setError(null);
            try {
                const result = await monitoringApi.updateDriftAlertDisposition(
                    alertId,
                    action,
                    actor,
                    note,
                );
                if (request !== actionRequest.current) return null;
                detailRequest.current += 1;
                setDetail(result);
                setLoading(false);
                setError(null);
                return result;
            } catch (err: unknown) {
                if (request === actionRequest.current) setError(dispositionError(err, action));
                return null;
            } finally {
                if (request === actionRequest.current) setActionPending(false);
            }
        },
        [alertId],
    );

    return { detail, loading, error, actionPending, refresh, applyDisposition };
}
