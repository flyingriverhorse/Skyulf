import { Dataset, IngestionJobResponse, IngestionStatus, DataSourceCreate } from '../types/api';

const API_BASE = '/data/api';
const INGESTION_BASE = '/api/ingestion';

export const DatasetService = {
  getAll: async (): Promise<Dataset[]> => {
    const response = await fetch(`${API_BASE}/sources`);
    if (!response.ok) {
      throw new Error('Failed to fetch datasets');
    }
    const data = await response.json();
    return data.sources || [];
  },

  getUsable: async (): Promise<Dataset[]> => {
    const response = await fetch(`${API_BASE}/sources/usable`);
    if (!response.ok) {
      throw new Error('Failed to fetch usable datasets');
    }
    const data = await response.json();
    return data.sources || [];
  },

  upload: async (file: File): Promise<IngestionJobResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${INGESTION_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to upload file');
    }
    
    return await response.json();
  },

  createSource: async (data: DataSourceCreate): Promise<IngestionJobResponse> => {
    const response = await fetch(`${INGESTION_BASE}/database`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create data source');
    }
    
    return await response.json();
  },

  getIngestionStatus: async (sourceId: string): Promise<IngestionStatus> => {
    const response = await fetch(`${INGESTION_BASE}/${sourceId}/status`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch ingestion status');
    }
    
    return await response.json();
  },

  cancelIngestion: async (sourceId: string): Promise<void> => {
    const response = await fetch(`${INGESTION_BASE}/${sourceId}/cancel`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to cancel ingestion');
    }
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
  },

  delete: async (id: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/sources/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete dataset');
    }
  },

  getProfile: async (id: string): Promise<any> => {
    const response = await fetch(`/api/pipeline/datasets/${id}/schema`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset profile');
    }
    const data = await response.json();
    
    // Transform backend response to match frontend expectation
    const columns = Object.values(data.columns).map((col: any) => ({
      name: col.name,
      dtype: col.dtype,
      missing_count: col.missing_count,
      missing_percentage: (col.missing_ratio || 0) * 100,
      distinct_count: col.unique_count,
      numeric_summary: col.column_type === 'numeric' ? {
        mean: col.mean_value,
        std: col.std_value,
        minimum: col.min_value,
        maximum: col.max_value
      } : undefined
    }));

    const totalCells = data.row_count * data.column_count;
    const missingCells = columns.reduce((acc, col) => acc + col.missing_count, 0);

    return {
      metrics: {
        row_count: data.row_count,
        column_count: data.column_count,
        missing_cells: missingCells,
        missing_percentage: totalCells > 0 ? (missingCells / totalCells * 100).toFixed(2) : 0
      },
      columns: columns
    };
  }
};
