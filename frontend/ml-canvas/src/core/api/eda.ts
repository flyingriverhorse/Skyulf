import axios from 'axios';

const API_BASE = '/api/eda';

export interface Filter {
    column: string;
    operator: '==' | '!=' | '>' | '<' | '>=' | '<=' | 'in';
    value: any;
}

export const EDAService = {
  analyze: async (datasetId: number, targetCol?: string, excludeCols?: string[], filters?: Filter[]) => {
    const response = await axios.post(`${API_BASE}/${datasetId}/analyze`, {
        target_col: targetCol,
        exclude_cols: excludeCols,
        filters: filters
    });
    return response.data;
  },

  getLatestReport: async (datasetId: number) => {
    const response = await axios.get(`${API_BASE}/${datasetId}/latest`);
    return response.data;
  },

  getReport: async (reportId: number) => {
    const response = await axios.get(`${API_BASE}/reports/${reportId}`);
    return response.data;
  },

  getHistory: async (datasetId: number) => {
    const response = await axios.get(`${API_BASE}/${datasetId}/history`);
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
