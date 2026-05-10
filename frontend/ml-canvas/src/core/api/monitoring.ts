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
}

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
}

export interface GroupedIssue {
    error_type: string;
    route: string;
    count: number;
    last_seen: string;
    first_seen: string;
    sample_id: number;
}

export interface SlowNodeAggregate {
    step_type: string;
    count: number;
    total_seconds: number;
    avg_seconds: number;
    p95_seconds: number;
    max_seconds: number;
    sample_node_id?: string | null;
}

export interface SlowNodesResponse {
    days: number;
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

    getDriftStatus: async (): Promise<DriftStatusSummary> => {
        const response = await apiClient.get<DriftStatusSummary>('/monitoring/drift/status');
        return response.data;
    },

    getErrors: async (limit = 100, since?: string, showResolved = false): Promise<ErrorEvent[]> => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (since) params.set('since', since);
        if (showResolved) params.set('show_resolved', 'true');
        const response = await apiClient.get<ErrorEvent[]>(`/monitoring/errors?${params}`);
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

    getPipelineLogs: async (limit = 200, since?: string, pipelineId?: string): Promise<PipelineRunLog[]> => {
        const params = new URLSearchParams({ limit: String(limit) });
        if (since) params.set('since', since);
        if (pipelineId) params.set('pipeline_id', pipelineId);
        const response = await apiClient.get<PipelineRunLog[]>(`/monitoring/pipeline-logs?${params}`);
        return response.data;
    },

    clearPipelineLogs: async (): Promise<void> => {
        await apiClient.delete('/monitoring/pipeline-logs');
    },
};
