import { apiClientV2, PipelineConfigModel } from './client';

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
  result: Record<string, any> | null;
  logs?: string[];
}

export interface RunPipelineRequest extends PipelineConfigModel {
  target_node_id?: string;
  job_type?: 'training' | 'tuning';
}

export interface RunPipelineResponse {
  message: string;
  pipeline_id: string;
  job_id: string;
}

export const jobsApi = {
  runPipeline: async (payload: RunPipelineRequest): Promise<RunPipelineResponse> => {
    const response = await apiClientV2.post<RunPipelineResponse>('/pipeline/run', payload);
    return response.data;
  },

  getJob: async (jobId: string): Promise<JobInfo> => {
    const response = await apiClientV2.get<JobInfo>(`/pipeline/jobs/${jobId}`);
    return response.data;
  },

  cancelJob: async (jobId: string): Promise<void> => {
    await apiClientV2.post(`/pipeline/jobs/${jobId}/cancel`);
  },

  listJobs: async (limit: number = 50): Promise<JobInfo[]> => {
    const response = await apiClientV2.get<JobInfo[]>('/pipeline/jobs', { params: { limit } });
    return response.data;
  },

  getHyperparameters: async (modelType: string): Promise<any[]> => {
    const response = await apiClientV2.get<any[]>(`/pipeline/hyperparameters/${modelType}`);
    return response.data;
  },

  getDefaultSearchSpace: async (modelType: string): Promise<Record<string, any>> => {
    const response = await apiClientV2.get<Record<string, any>>(`/pipeline/hyperparameters/${modelType}/defaults`);
    return response.data;
  },

  getLatestTuningJob: async (nodeId: string): Promise<JobInfo | null> => {
    const response = await apiClientV2.get<JobInfo | null>(`/pipeline/jobs/tuning/latest/${nodeId}`);
    return response.data;
  },

  getBestTuningJobForModel: async (modelType: string): Promise<JobInfo | null> => {
    const response = await apiClientV2.get<JobInfo | null>(`/pipeline/jobs/tuning/best/${modelType}`);
    return response.data;
  },

  getTuningJobsForModel: async (modelType: string): Promise<JobInfo[]> => {
    const response = await apiClientV2.get<JobInfo[]>(`/pipeline/jobs/tuning/history/${modelType}`);
    return response.data;
  }
};
