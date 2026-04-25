import axios from 'axios';
import type { EDAProfile } from '../types/edaProfile';

const API_BASE = '/api/eda';

export interface Filter {
    column: string;
    operator: '==' | '!=' | '>' | '<' | '>=' | '<=' | 'in';
    value: string | number | boolean | Array<string | number>;
}

export interface EDAReport {
    id?: number;
    status?: 'PENDING' | 'COMPLETED' | 'FAILED' | string;
    profile_data?: EDAProfile & {
        target_col?: string;
        excluded_columns?: string[];
        task_type?: string;
    };
    created_at?: string;
    error?: string | null;
    error_message?: string | null;
    [extra: string]: unknown;
}

export interface EDAHistoryEntry {
    id: number;
    status: string;
    created_at: string;
    target_col?: string;
    description?: string;
    [extra: string]: unknown;
}

export const EDAService = {
  analyze: async (datasetId: number, targetCol?: string, excludeCols?: string[], filters?: Filter[], taskType?: string) => {
    const response = await axios.post(`${API_BASE}/${datasetId}/analyze`, {
        target_col: targetCol,
        exclude_cols: excludeCols,
        filters: filters,
        task_type: taskType
    });
    return response.data;
  },

  getLatestReport: async (datasetId: number): Promise<EDAReport> => {
    const response = await axios.get<EDAReport>(`${API_BASE}/${datasetId}/latest`);
    return response.data;
  },

  getReport: async (reportId: number): Promise<EDAReport> => {
    const response = await axios.get<EDAReport>(`${API_BASE}/reports/${reportId}`);
    return response.data;
  },

  cancelJob: async (reportId: number) => {
    const response = await axios.post(`${API_BASE}/reports/${reportId}/cancel`);
    return response.data;
  },

  getHistory: async (datasetId: number): Promise<EDAHistoryEntry[]> => {
    const response = await axios.get<EDAHistoryEntry[]>(`${API_BASE}/${datasetId}/history`);
    return response.data;
  },

  getDecomposition: async (datasetId: number, measureCol: string | null, measureAgg: string, splitCol: string, filters: Filter[]) => {
    const response = await axios.post(`${API_BASE}/${datasetId}/decomposition`, {
        measure_col: measureCol,
        measure_agg: measureAgg,
        split_col: splitCol,
        filters: filters
    });
    return response.data;
  }
};
