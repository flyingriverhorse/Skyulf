export interface Dataset {
  id: string;
  source_id?: string;
  name: string;
  description?: string;
  created_at: string;
  rows?: number;
  columns?: number;
  size_bytes?: number;
  format?: string;
  source_metadata?: unknown;
}

export interface IngestionJobResponse {
  job_id: string;
  status: string;
  message: string;
  file_id?: string;
}

export interface IngestionStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  details?: unknown;
  updated_at: string;
}

export interface DataSourceCreate {
  name: string;
  type: string;
  config: Record<string, unknown>;
  description?: string;
}
