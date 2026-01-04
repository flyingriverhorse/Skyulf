import { apiClient, PipelineConfigModel } from './client';
import axios from 'axios';

export type JobStatus = 'queued' | 'running' | 'completed' | 'succeeded' | 'failed' | 'cancelled' | 'pending';

export interface JobInfo {
  job_id: string;
  pipeline_id: string;
  node_id: string;
  dataset_id?: string;
  dataset_name?: string;
  job_type: 'training' | 'tuning' | 'eda' | 'ingestion';
  status: JobStatus;
  start_time: string | null;
  end_time: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
  logs?: string[];
  
  // Extended fields
  model_type?: string;
  hyperparameters?: Record<string, unknown>;
  created_at: string;
  metrics?: Record<string, number>;
  config?: unknown;
  search_strategy?: string;
  target_column?: string;
  dropped_columns?: string[];
  version?: number;
}

export interface RunPipelineRequest extends PipelineConfigModel {
  target_node_id?: string;
  job_type?: 'training' | 'tuning' | 'preview';
}

export interface RunPipelineResponse {
  message: string;
  pipeline_id: string;
  job_id: string;
}

export const jobsApi = {
  runPipeline: async (payload: RunPipelineRequest): Promise<RunPipelineResponse> => {
    const response = await apiClient.post<RunPipelineResponse>('/pipeline/run', payload);
    return response.data;
  },

  getJob: async (jobId: string): Promise<JobInfo> => {
    const response = await apiClient.get<JobInfo>(`/pipeline/jobs/${jobId}`);
    return response.data;
  },

  cancelJob: async (jobId: string): Promise<void> => {
    await apiClient.post(`/pipeline/jobs/${jobId}/cancel`);
  },

  getJobs: async (limit: number = 10, skip: number = 0, type?: 'training' | 'tuning'): Promise<JobInfo[]> => {
    const params: Record<string, unknown> = { limit, skip };
    if (type) {
      params.job_type = type;
    }
    const response = await apiClient.get<JobInfo[]>('/pipeline/jobs', { params });
    return response.data;
  },

  getEDAJobs: async (limit: number = 50): Promise<JobInfo[]> => {
    const response = await apiClient.get<any[]>('/eda/jobs/all', { params: { limit } });
    return response.data.map(job => ({
      job_id: String(job.id),
      pipeline_id: 'eda',
      node_id: 'eda',
      dataset_id: String(job.dataset_id),
      dataset_name: job.dataset_name,
      job_type: 'eda',
      status: job.status.toLowerCase() as JobStatus,
      start_time: job.created_at,
      end_time: job.updated_at || job.created_at,
      error: job.error,
      result: null,
      created_at: job.created_at,
      target_column: job.target_col
    }));
  },

  getIngestionJobs: async (limit: number = 50, skip: number = 0): Promise<JobInfo[]> => {
    // Use direct axios call to avoid /api prefix since data sources are at /data/api
    const response = await axios.get<any>('/data/api/sources', { params: { limit, skip } });
    return response.data.sources.map((source: any) => ({
      job_id: String(source.id),
      pipeline_id: 'ingestion',
      node_id: 'ingestion',
      dataset_id: String(source.id),
      dataset_name: source.name,
      job_type: 'ingestion',
      status: (source.test_status === 'success' ? 'succeeded' : source.test_status === 'failed' ? 'failed' : 'completed') as JobStatus,
      start_time: source.created_at,
      end_time: source.updated_at,
      error: null,
      result: null,
      created_at: source.created_at,
      model_type: source.type
    }));
  },

  getHyperparameters: async (modelType: string): Promise<unknown[]> => {
    const response = await apiClient.get<unknown[]>(`/pipeline/hyperparameters/${modelType}`);
    return response.data;
  },

  getDefaultSearchSpace: async (modelType: string): Promise<Record<string, unknown>> => {
    const response = await apiClient.get<Record<string, unknown>>(`/pipeline/hyperparameters/${modelType}/defaults`);
    return response.data;
  },

  getLatestTuningJob: async (nodeId: string): Promise<JobInfo | null> => {
    const response = await apiClient.get<JobInfo | null>(`/pipeline/jobs/tuning/latest/${nodeId}`);
    return response.data;
  },

  getBestTuningJob: async (modelType: string): Promise<JobInfo | null> => {
    const response = await apiClient.get<JobInfo | null>(`/pipeline/jobs/tuning/best/${modelType}`);
    return response.data;
  },

  getTuningHistory: async (modelType: string): Promise<JobInfo[]> => {
    const response = await apiClient.get<JobInfo[]>(`/pipeline/jobs/tuning/history/${modelType}`);
    return response.data;
  }
};
