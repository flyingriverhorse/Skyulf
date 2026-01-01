import axios from 'axios';

const API_BASE = '/api/eda';

export const EDAService = {
  analyze: async (datasetId: number, targetCol?: string) => {
    const response = await axios.post(`${API_BASE}/${datasetId}/analyze`, {
        target_col: targetCol
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
  }
};
