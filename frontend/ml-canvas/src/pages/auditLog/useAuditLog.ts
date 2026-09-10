import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useUsableDatasets } from '../../core/hooks/useDatasets';
import { pipelineVersionsApi, type AuditLogResponse } from '../../core/api/pipelineVersions';
import { toast } from '../../core/toast';
import {
    auditRequestFilters, describeAuditHistory, resolveDatasetSourceId, summarizeAuditHistory,
} from './auditHistoryModel';

/** Own the page's filters, request generation, and last successful response. */
export function useAuditLog() {
    const { data: datasets, isLoading: datasetsLoading } = useUsableDatasets();
    const [datasetId, setDatasetId] = useState<string>('');
    const [limit, setLimit] = useState<number>(50);
    const [actorFilter, setActorFilter] = useState<string>('all');
    const [kindFilter, setKindFilter] = useState<string>('all');
    const [fromTime, setFromTime] = useState<string>('');
    const [toTime, setToTime] = useState<string>('');
    const [data, setData] = useState<AuditLogResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Auto-pick the first dataset once the list arrives so the page renders
    // something useful on first load instead of an empty picker.
    useEffect(() => {
        if (!datasetId && datasets && datasets.length > 0) {
            const first = datasets[0];
            if (first) setDatasetId(resolveDatasetSourceId(first));
        }
    }, [datasets, datasetId]);

    // Track in-flight request id so a slow response doesn't clobber the
    // state set by a more recent one (e.g. rapid `datasetId`/`limit` toggles).
    const requestIdRef = useRef(0);

    const load = useCallback(async () => {
        if (!datasetId) return;
        const myRequestId = ++requestIdRef.current;
        setIsLoading(true);
        setError(null);
        try {
            const resp = await pipelineVersionsApi.audit(
                datasetId, limit,
                auditRequestFilters({ actorFilter, kindFilter, fromTime, toTime }),
            );
            if (myRequestId !== requestIdRef.current) return;
            setData(resp);
        } catch (e) {
            if (myRequestId !== requestIdRef.current) return;
            const msg = (e as Error).message || 'Failed to load audit trail';
            setError(msg);
            toast.error(msg);
        } finally {
            if (myRequestId === requestIdRef.current) setIsLoading(false);
        }
    }, [datasetId, limit, actorFilter, kindFilter, fromTime, toTime]);

    useEffect(() => {
        void load();
    }, [load]);

    const summary = useMemo(() => summarizeAuditHistory(data), [data]);

    // Facets come from the server's pre-filter pass, so selecting one actor
    // never removes the other actors from the dropdown.
    const actorOptions = useMemo(
        () => ({
            hasAnonymous: data?.facets?.has_anonymous_actor ?? false,
            userIds: data?.facets?.actors ?? [],
        }),
        [data],
    );

    const kindOptions = useMemo(() => data?.facets?.kinds ?? [], [data]);

    const datasetLabel = useMemo(() => {
        const match = datasets?.find(d => resolveDatasetSourceId(d) === datasetId);
        return match?.name ?? 'this dataset';
    }, [datasets, datasetId]);

    const history = describeAuditHistory(data, datasetLabel, limit, {
        actorFilter, kindFilter, fromTime, toTime,
    });
    return {
        datasets, datasetsLoading, datasetId, setDatasetId, limit, setLimit,
        actorFilter, setActorFilter, kindFilter, setKindFilter, fromTime, setFromTime,
        toTime, setToTime, data, isLoading, error, load, summary, actorOptions, kindOptions,
        ...history,
    };
}

export type AuditLogState = ReturnType<typeof useAuditLog>;
