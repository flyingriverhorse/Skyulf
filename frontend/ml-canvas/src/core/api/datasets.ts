import { Dataset } from '../types/api';

const API_BASE = '/data/api';

export const DatasetService = {
  getAll: async (): Promise<Dataset[]> => {
    const response = await fetch(`${API_BASE}/sources`);
    if (!response.ok) {
      throw new Error('Failed to fetch datasets');
    }
    const data = await response.json();
    return data.sources || [];
  },

  getById: async (id: string): Promise<Dataset> => {
    const response = await fetch(`${API_BASE}/sources/${id}`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset');
    }
    const data = await response.json();
    return data.source;
  },

  getSample: async (id: string, limit: number = 1): Promise<any[]> => {
    const response = await fetch(`${API_BASE}/sources/${id}/sample?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset sample');
    }
    const data = await response.json();
    return data.data;
  }
};
