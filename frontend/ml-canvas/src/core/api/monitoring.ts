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
}

export interface DriftJobOption {
    job_id: string;
    dataset_name: string;
    filename: string;
    created_at?: string;
}

export const monitoringApi = {
    getJobs: async (): Promise<DriftJobOption[]> => {
        const response = await apiClient.get<DriftJobOption[]>('/monitoring/jobs');
        return response.data;
    },

    calculateDrift: async (jobId: string, file: File, datasetName?: string): Promise<DriftReport> => {
        const formData = new FormData();
        formData.append('job_id', jobId);
        formData.append('file', file);
        if (datasetName) {
            formData.append('dataset_name', datasetName);
        }

        const response = await apiClient.post<DriftReport>('/monitoring/drift/calculate', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },
};
