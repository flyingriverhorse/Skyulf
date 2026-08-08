import { apiClient } from './client';

export interface DriftMetric {
    metric: string;
    value: number;
    has_drift: boolean;
    threshold: number;
}

export interface DriftBin {
    bin_start: number;
    bin_end: number;
    reference_count: number;
    current_count: number;
}

export interface DriftDistribution {
    bins: DriftBin[];
}

export interface ColumnDrift {
    column: string;
    metrics: DriftMetric[];
    drift_detected: boolean;
    suggestions: string[];
    distribution?: DriftDistribution;
}

export interface DriftReport {
    reference_rows: number;
    current_rows: number;
    drifted_columns_count: number;
    column_drifts: Record<string, ColumnDrift>;
    missing_columns: string[];
    new_columns: string[];
    feature_importances?: Record<string, number>;
    alert_id?: number | null;
    severity: DriftAlertSeverity;
    threshold_version?: number | null;
    deployment_id?: number | null;
    model_version?: string | null;
}

export interface DriftHistoryEntry {
    id: number;
    job_id: string;
    dataset_name?: string;
    reference_rows?: number;
    current_rows?: number;
    drifted_columns_count?: number;
    total_columns?: number;
    summary?: Record<string, { drifted: boolean; psi?: number; wasserstein?: number; ks_p_value?: number }>;
    created_at?: string;
    severity: DriftAlertSeverity;
    status: DriftAlertStatus;
    owner?: string | null;
    acknowledged_at?: string | null;
    resolved_at?: string | null;
    threshold_version?: number | null;
    threshold_psi?: number | null;
    threshold_ks?: number | null;
    threshold_wasserstein?: number | null;
    threshold_kl?: number | null;
    deployment_id?: number | null;
    model_version?: string | null;
    evaluation_status: DriftEvaluationStatus;
    error_message?: string | null;
}

/** Triage severity derived server-side at evaluation time (`_classify_drift_severity`). */
export type DriftAlertSeverity = 'none' | 'warning' | 'critical';

/** Disposition lifecycle: new -> acknowledged -> (resolved | reopened) -> ... */
export type DriftAlertStatus = 'new' | 'acknowledged' | 'resolved' | 'reopened';

/** Distinguishes a completed check from an explicit no-baseline/failed evaluation outcome. */
export type DriftEvaluationStatus = 'completed' | 'no_baseline' | 'failed';

export type DriftDispositionAction = 'acknowledge' | 'resolve' | 'reopen';

export interface DriftDispositionEntry {
    status: DriftAlertStatus;
    actor: string;
    note: string | null;
    at: string;
}

export interface DriftAlertDetail extends DriftHistoryEntry {
    column_drifts?: Record<string, ColumnDrift> | null;
    disposition_history: DriftDispositionEntry[];
}

export interface DriftJobOption {
    job_id: string;
    dataset_name: string;
    filename: string;
    created_at?: string;
    model_type?: string;
    target_column?: string;
    n_features?: number;
    n_rows?: number;
    description?: string;
    best_metric?: string;
}

export interface DriftThresholds {
    psi?: number;
    ks?: number;
    wasserstein?: number;
    kl?: number;
}

export interface DriftStatusSummary {
    has_drift: boolean;
    drifted_jobs: number;
    latest_check?: string;
    unacknowledged_critical: number;
}

/** Typed severity derived server-side from `status_code` (see `_classify_error_severity`). */
export type ErrorSeverity = 'critical' | 'warning' | 'info';

export interface ErrorEvent {
    id: number;
    route: string;
    error_type: string;
    message: string;
    traceback?: string;
    job_id?: string;
    status_code: number;
    created_at: string;
    resolved_at?: string | null;
    severity: ErrorSeverity;
}

/** Every typed facet value present across the full unfiltered error history. */
export interface ErrorEventFacets {
    severities: ErrorSeverity[];
    error_types: string[];
    job_ids: string[];
}

/** Server-side error search filters. Omitted/empty fields are not sent. */
export interface ErrorEventFilters {
    severity?: ErrorSeverity;
    errorType?: string;
    jobId?: string;
    q?: string;
}

export interface ErrorEventSearchResponse {
    total: number;
    /** History size before filters — lets the UI say "3 of 120" vs. "no history". */
    total_unfiltered: number;
    facets: ErrorEventFacets;
    filters: {
        since: string | null;
        show_resolved: boolean;
        severity: string | null;
        error_type: string | null;
        job_id: string | null;
        q: string | null;
    };
    entries: ErrorEvent[];
}

/** Every typed facet value present across the full unfiltered pipeline log history. */
export interface PipelineLogFacets {
    levels: string[];
    node_types: string[];
    pipeline_ids: string[];
    node_ids: string[];
}

/** Server-side pipeline log search filters. Omitted/empty fields are not sent. */
export interface PipelineLogFilters {
    level?: string;
    nodeType?: string;
    nodeId?: string;
    q?: string;
}

export interface PipelineLogSearchResponse {
    total: number;
    total_unfiltered: number;
    facets: PipelineLogFacets;
    filters: {
        since: string | null;
        pipeline_id: string | null;
        level: string | null;
        node_type: string | null;
        node_id: string | null;
        q: string | null;
    };
    entries: PipelineRunLog[];
}

export interface GroupedIssue {
    error_type: string;
    route: string;
    count: number;
    last_seen: string;
    first_seen: string;
    sample_id: number;
}

export interface SlowNodeRun {
    job_id: string;
    pipeline_id: string;
    node_id: string;
    dataset_source_id: string;
    execution_seconds: number;
    finished_at?: string | null;
    is_outlier: boolean;
}

export interface SlowNodeAggregate {
    step_type: string;
    count: number;
    total_seconds: number;
    avg_seconds: number;
    p95_seconds: number;
    max_seconds: number;
    sample_node_id?: string | null;
    is_single_run: boolean;
    sample_is_representative: boolean;
    contributing_runs: SlowNodeRun[];
}

export interface SlowNodesResponse {
    days: number;
    unit: string;
    total_jobs_scanned: number;
    total_node_runs: number;
    aggregates: SlowNodeAggregate[];
}

export interface PipelineLogEntry {
    node_id?: string | null;
    node_type?: string | null;
    level: string;
    logger?: string | null;
    message: string;
}

export interface PipelineRunLog {
    id: number;
    pipeline_id?: string | null;
    node_id?: string | null;
    node_type?: string | null;
    level: string;
    logger?: string | null;
    message: string;
    run_at?: string | null;
}

export const monitoringApi = {
    getJobs: async (): Promise<DriftJobOption[]> => {
        const response = await apiClient.get<DriftJobOption[]>('/monitoring/jobs');
        return response.data;
    },

    calculateDrift: async (jobId: string, file: File, datasetName?: string, thresholds?: DriftThresholds): Promise<DriftReport> => {
        const formData = new FormData();
        formData.append('job_id', jobId);
        formData.append('file', file);
        if (datasetName) {
            formData.append('dataset_name', datasetName);
        }
        if (thresholds?.psi != null) formData.append('threshold_psi', String(thresholds.psi));
        if (thresholds?.ks != null) formData.append('threshold_ks', String(thresholds.ks));
        if (thresholds?.wasserstein != null) formData.append('threshold_wasserstein', String(thresholds.wasserstein));
        if (thresholds?.kl != null) formData.append('threshold_kl', String(thresholds.kl));

        const response = await apiClient.post<DriftReport>('/monitoring/drift/calculate', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    updateJobDescription: async (jobId: string, description: string): Promise<void> => {
        await apiClient.patch(`/monitoring/jobs/${jobId}/description`, { description });
    },

    getDriftHistory: async (jobId: string): Promise<DriftHistoryEntry[]> => {
        const response = await apiClient.get<DriftHistoryEntry[]>(`/monitoring/drift/history/${jobId}`);
        return response.data;
    },

    getDriftAlert: async (alertId: number): Promise<DriftAlertDetail> => {
        const response = await apiClient.get<DriftAlertDetail>(`/monitoring/drift/alerts/${alertId}`);
        return response.data;
    },

    updateDriftAlertDisposition: async (
        alertId: number,
        action: DriftDispositionAction,
        actor: string,
        note?: string,
    ): Promise<DriftAlertDetail> => {
        const response = await apiClient.patch<DriftAlertDetail>(
            `/monitoring/drift/alerts/${alertId}/disposition`,
            { action, actor, note: note ?? null },
        );
        return response.data;
    },

    getDriftStatus: async (): Promise<DriftStatusSummary> => {
        const response = await apiClient.get<DriftStatusSummary>('/monitoring/drift/status');
        return response.data;
    },

    getErrors: async (
        limit = 100,
        since?: string,
        showResolved = false,
        filters: ErrorEventFilters = {},
    ): Promise<ErrorEventSearchResponse> => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (since) params.set('since', since);
        if (showResolved) params.set('show_resolved', 'true');
        if (filters.severity) params.set('severity', filters.severity);
        if (filters.errorType) params.set('error_type', filters.errorType);
        if (filters.jobId) params.set('job_id', filters.jobId);
        if (filters.q) params.set('q', filters.q);
        const response = await apiClient.get<ErrorEventSearchResponse>(`/monitoring/errors?${params}`);
        return response.data;
    },

    getUnresolvedCount: async (): Promise<number> => {
        const response = await apiClient.get<{ count: number }>('/monitoring/errors/count');
        return response.data.count;
    },

    getTimeline: async (hours = 24): Promise<{ hour: string; count: number }[]> => {
        const response = await apiClient.get<{ hour: string; count: number }[]>(
            `/monitoring/errors/timeline?hours=${hours}`
        );
        return response.data;
    },

    getError: async (id: number): Promise<ErrorEvent> => {
        const response = await apiClient.get<ErrorEvent>(`/monitoring/errors/${id}`);
        return response.data;
    },

    resolveError: async (id: number): Promise<ErrorEvent> => {
        const response = await apiClient.patch<ErrorEvent>(`/monitoring/errors/${id}/resolve`);
        return response.data;
    },

    unresolveError: async (id: number): Promise<ErrorEvent> => {
        const response = await apiClient.patch<ErrorEvent>(`/monitoring/errors/${id}/unresolve`);
        return response.data;
    },

    clearErrors: async (): Promise<{ deleted: number }> => {
        const response = await apiClient.delete<{ deleted: number }>('/monitoring/errors');
        return response.data;
    },

    getGrouped: async (): Promise<GroupedIssue[]> => {
        const response = await apiClient.get<GroupedIssue[]>('/monitoring/errors/grouped');
        return response.data;
    },

    getSlowNodes: async (days = 7, limit = 10): Promise<SlowNodesResponse> => {
        const response = await apiClient.get<SlowNodesResponse>(
            `/monitoring/slow-nodes?days=${days}&limit=${limit}`,
        );
        return response.data;
    },

    // ── Pipeline run logs ──────────────────────────────────────────────
    logPipelineRun: async (pipelineId: string | null, entries: PipelineLogEntry[]): Promise<void> => {
        if (entries.length === 0) return;
        await apiClient.post('/monitoring/pipeline-logs', { pipeline_id: pipelineId, entries });
    },

    getPipelineLogs: async (
        limit = 200,
        since?: string,
        pipelineId?: string,
        filters: PipelineLogFilters = {},
    ): Promise<PipelineLogSearchResponse> => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (since) params.set('since', since);
        if (pipelineId) params.set('pipeline_id', pipelineId);
        if (filters.level) params.set('level', filters.level);
        if (filters.nodeType) params.set('node_type', filters.nodeType);
        if (filters.nodeId) params.set('node_id', filters.nodeId);
        if (filters.q) params.set('q', filters.q);
        const response = await apiClient.get<PipelineLogSearchResponse>(`/monitoring/pipeline-logs?${params}`);
        return response.data;
    },

    clearPipelineLogs: async (): Promise<void> => {
        await apiClient.delete('/monitoring/pipeline-logs');
    },
};
