import { Dataset, IngestionJobResponse, IngestionStatus, DataSourceCreate } from '../types/api';

const API_BASE = '/data/api';
const INGESTION_BASE = '/api/ingestion';

/** Fetch error carrying the HTTP status, so callers can distinguish e.g. a
 * 404 (dataset no longer exists) from a transient network/server failure. */
export class DatasetApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = 'DatasetApiError';
  }
}

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

  /**
   * Same as `upload`, but reports real upload progress (0-100) via
   * `onProgress`. Uses XHR because `fetch` has no upload-progress API.
   */
  uploadWithProgress: (file: File, onProgress: (percent: number) => void): Promise<IngestionJobResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    return new Promise<IngestionJobResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${INGESTION_BASE}/upload`);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };

      xhr.onload = () => {
        let payload: unknown;
        try {
          payload = JSON.parse(xhr.responseText);
        } catch {
          payload = undefined;
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload as IngestionJobResponse);
        } else {
          const detail = (payload as { detail?: string } | undefined)?.detail;
          reject(new Error(detail || 'Failed to upload file'));
        }
      };

      xhr.onerror = () => reject(new Error('Failed to upload file'));
      xhr.onabort = () => reject(new Error('Upload cancelled'));

      xhr.send(formData);
    });
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
    // nosemgrep: node-ssrf -- browser fetch to our own fixed API_BASE; `id` is
    // an encoded path segment (dataset id), never an attacker-supplied host/URL.
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
      throw new DatasetApiError('Failed to fetch dataset sample', response.status);
    }
    const data = await response.json();
    return data.data;
  },

  // UI-friendly dataset profiling (schema + basic stats)
  getProfile: async (id: string): Promise<DatasetProfile> => {
    const response = await fetch(`/api/pipeline/datasets/${encodeURIComponent(id)}/schema`);
    if (!response.ok) {
      throw new DatasetApiError('Failed to fetch dataset profile', response.status);
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

  exportData: async (id: string, format: 'csv' | 'parquet' = 'csv', limit: number = 10000): Promise<void> => {
    const params = new URLSearchParams({ format, limit: limit.toString() });
    // nosemgrep: node-ssrf -- browser fetch to our own fixed API_BASE; `id` is
    // an encoded path segment (dataset id), never an attacker-supplied host/URL.
    const response = await fetch(`${API_BASE}/sources/${encodeURIComponent(id)}/export?${params.toString()}`);
    if (!response.ok) {
      throw new Error('Failed to export dataset');
    }
    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition') || '';
    const filenameMatch = disposition.match(/filename=(.+)/);
    const filename = filenameMatch?.[1] ?? `export.${format}`;

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

};

export interface DatasetProfile {
  metrics: {
    row_count: number;
    column_count: number;
    missing_cells: number;
    missing_percentage: number;
  };
  correlations?: {
    columns: string[];
    values: number[][];
  };
  correlations_with_target?: {
    columns: string[];
    values: number[][];
  };
  vif?: Record<string, number>;
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
    } | undefined;
    text_summary?: {
      sentiment_distribution?: Record<string, number>;
    } | undefined;
  }>;
}
