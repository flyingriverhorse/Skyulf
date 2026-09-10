import type { AuditLogResponse } from '../../core/api/pipelineVersions';

/** Resolve the dataset_source_id used by the pipelines API. The canvas
 *  Toolbar saves under `Dataset.id` (the value the dataset dropdown
 *  binds to via `option.value={d.id}`), so the audit picker must use
 *  the same id. We coerce to string because some legacy `Dataset` rows
 *  carry a numeric id that breaks `String.prototype.slice` downstream. */
export const resolveDatasetSourceId = (d: { id: string | number }): string => String(d.id);

export interface AuditFilterState {
    actorFilter: string;
    kindFilter: string;
    fromTime: string;
    toTime: string;
}

/** Match the API's omission semantics without converting datetime-local values. */
export function auditRequestFilters({ actorFilter, kindFilter, fromTime, toTime }: AuditFilterState) {
    return {
        ...(actorFilter !== 'all' ? { actor: actorFilter } : {}),
        ...(kindFilter !== 'all' ? { kind: kindFilter } : {}),
        ...(fromTime ? { createdAfter: fromTime } : {}),
        ...(toTime ? { createdBefore: toTime } : {}),
    };
}

/** Summaries count the entries in the current response. */
export function summarizeAuditHistory(data: AuditLogResponse | null) {
    if (!data) return null;
    const entries = data.entries;
    const uniqueUsers = new Set(
        entries.map(e => (e.user_id !== null ? `u${e.user_id}` : 'anon')),
    );
    let added = 0;
    let removed = 0;
    let modified = 0;
    for (const e of entries) {
        added += e.diff.nodes_added.length;
        removed += e.diff.nodes_removed.length;
        modified += e.diff.nodes_modified.length;
    }
    return {
        saves: entries.length,
        users: uniqueUsers.size,
        added,
        removed,
        modified,
    };
}

/** Describe the server's full-history scope and the current page window. */
export function describeAuditHistory(
    data: AuditLogResponse | null,
    datasetLabel: string,
    limit: number,
    filters: AuditFilterState,
) {
    const { actorFilter, kindFilter, fromTime, toTime } = filters;
    const filteredEntries = data?.entries ?? [];

    const hasFilters =
        actorFilter !== 'all' || kindFilter !== 'all' || fromTime !== '' || toTime !== '';
    const matchingTotal = data?.total ?? 0;
    const historyTotal = data?.total_unfiltered ?? 0;
    const visibleCount = filteredEntries.length;
    const limitLabel = `${limit}`;
    const historyScopeText = data
        ? hasFilters
            ? `Showing ${visibleCount} of ${matchingTotal} matching saves for ${datasetLabel}. History total ${historyTotal}.`
            : `Showing ${visibleCount} of ${historyTotal} saves for ${datasetLabel}.`
        : null;
    const historyMetaText = data
        ? `Page limit ${limitLabel}. Newest first. Retention is not reported by the API. Filters are applied across the full history, not just this page.`
        : null;
    const emptyStateText = describeEmptyHistory(data, visibleCount, historyTotal, datasetLabel, limitLabel);
    return { filteredEntries, historyScopeText, historyMetaText, emptyStateText };
}

/** Distinguish absent history from a history with no matching records. */
function describeEmptyHistory(
    data: AuditLogResponse | null,
    visibleCount: number,
    historyTotal: number,
    datasetLabel: string,
    limitLabel: string,
) {
    return data
        ? visibleCount === 0
            ? historyTotal === 0
                ? `No saves recorded for ${datasetLabel} yet. The audit API shows the newest ${limitLabel} saves first, but this dataset currently has no history. Retention is not reported by the API.`
                : `No audit records match the current filters for ${datasetLabel}. Filters were applied across all ${historyTotal} saves, so widening them is the only way to see more. Retention is not reported by the API.`
            : null
        : null;
}
