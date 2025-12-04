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
    return response.json();
  }
};
