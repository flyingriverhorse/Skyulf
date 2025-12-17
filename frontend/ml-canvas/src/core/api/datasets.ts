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
    const response = await fetch(`${INGESTION_BASE}/${encodeURIComponent(sourceId)}/status`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch ingestion status');
    }
    
    return await response.json();
  },

  cancelIngestion: async (sourceId: string): Promise<void> => {
    const response = await fetch(`${INGESTION_BASE}/${encodeURIComponent(sourceId)}/cancel`, {
      method: 'POST',
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to cancel ingestion');
    }
  },

  getById: async (id: string): Promise<Dataset> => {
    const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(id)}`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset');
    }
    const data = await response.json();
    return data.source;
  },

  getSample: async (id: string, limit: number = 1): Promise<unknown[]> => {
    const params = new URLSearchParams({ limit: limit.toString() });
    const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(id)}/sample?${params.toString()}`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset sample');
    }
    const data = await response.json();
    return data.data;
  },

  // UI-friendly dataset profiling (schema + basic stats)
  getProfile: async (id: string): Promise<DatasetProfile> => {
    const response = await fetch(`/api/pipeline/datasets/${encodeURIComponent(id)}/schema`);
    if (!response.ok) {
      throw new Error('Failed to fetch dataset profile');
    }
    const data = await response.json();
    
    // Transform backend response to match frontend expectation
    const columns = Object.values(data.columns).map((col: unknown) => {
      const c = col as Record<string, unknown>;
      return {
        name: c.name as string,
        dtype: c.dtype as string,
        missing_count: (c.missing_count as number) || 0,
        missing_percentage: ((c.missing_ratio as number) || 0) * 100,
        distinct_count: (c.unique_count as number) || 0,
        numeric_summary: c.column_type === 'numeric' ? {
          mean: c.mean_value as number,
          std: c.std_value as number,
          minimum: c.min_value as number,
          maximum: c.max_value as number
        } : undefined
      };
    });

    const totalCells = (data.row_count as number) * (data.column_count as number);
    const missingCells = columns.reduce((acc, col) => acc + (col.missing_count as number), 0);

    return {
      metrics: {
        row_count: data.row_count,
        column_count: data.column_count,
        missing_cells: missingCells,
        missing_percentage: totalCells > 0 ? Number((missingCells / totalCells * 100).toFixed(2)) : 0
      },
      columns: columns
    };
  },

  delete: async (id: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete dataset');
    }
  },

};

export interface DatasetProfile {
  metrics: {
    row_count: number;
    column_count: number;
    missing_cells: number;
    missing_percentage: number;
  };
  columns: Array<{
    name: string;
    dtype: string;
    missing_count: number;
    missing_percentage: number;
    distinct_count: number;
    numeric_summary?: {
      mean: number;
      std: number;
      minimum: number;
      maximum: number;
    };
  }>;
};
