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
};
