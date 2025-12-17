import { apiClient, PipelineConfigModel } from './client';

export type JobStatus = 'queued' | 'running' | 'completed' | 'succeeded' | 'failed' | 'cancelled';

export interface JobInfo {
  job_id: string;
  pipeline_id: string;
  node_id: string;
  dataset_id?: string;
  dataset_name?: string;
  job_type: 'training' | 'tuning';
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
    const params: unknown = { limit, skip };
    if (type) {
      (params as any).job_type = type;
    }
    const response = await apiClient.get<JobInfo[]>('/pipeline/jobs', { params: params as any });
    return response.data;
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
