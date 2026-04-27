import { apiClient, PipelineConfigModel } from './client';
import axios from 'axios';

export type JobStatus = 'queued' | 'running' | 'completed' | 'succeeded' | 'failed' | 'cancelled' | 'pending';

export interface JobInfo {
  job_id: string;
  pipeline_id: string;
  node_id: string;
  dataset_id?: string;
  dataset_name?: string;
  job_type: 'basic_training' | 'advanced_tuning' | 'eda' | 'ingestion';
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
  graph?: Record<string, unknown>;
  search_strategy?: string;
  target_column?: string;
  dropped_columns?: string[];
  version?: number;
  promoted_at?: string | null;

  // Parallel branch metadata
  branch_index?: number | null;
  parent_pipeline_id?: string | null;
}

export interface RunPipelineRequest extends PipelineConfigModel {
  target_node_id?: string;
  job_type?: 'basic_training' | 'advanced_tuning' | 'preview';
}

// One entry per completed branch in the latest run group of a trainer
// node. Single-terminal (merge) runs return exactly one entry with
// `branch_index = null`; parallel runs return one per branch.
export interface NodeSummaryEntry {
  summary: string;
  branch_index: number | null;
  pipeline_id: string;
  parent_pipeline_id: string | null;
  finished_at: string | null;
}

export interface RunPipelineResponse {
  message: string;
  pipeline_id: string;
  job_id: string;
  job_ids: string[];  // All job IDs when parallel branches are detected
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
    interface RawEDAJob {
      id: string | number;
      dataset_id: string | number;
      dataset_name: string;
      status: string;
      created_at: string;
      updated_at?: string;
      error?: string | null;
      target_col?: string;
    }
    const response = await apiClient.get<RawEDAJob[]>('/eda/jobs/all', { params: { limit } });
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
      error: job.error ?? null,
      result: null,
      created_at: job.created_at,
      ...(job.target_col !== undefined ? { target_column: job.target_col } : {})
    }));
  },

  getIngestionJobs: async (limit: number = 50, skip: number = 0): Promise<JobInfo[]> => {
    // Use direct axios call to avoid /api prefix since data sources are at /data/api
    interface RawSource {
      id: string | number;
      name: string;
      type: string;
      test_status?: string;
      created_at: string;
      updated_at?: string;
    }
    const response = await axios.get<{ sources: RawSource[] }>('/data/api/sources', { params: { limit, skip } });
    return response.data.sources.map((source) => ({
      job_id: String(source.id),
      pipeline_id: 'ingestion',
      node_id: 'ingestion',
      dataset_id: String(source.id),
      dataset_name: source.name,
      job_type: 'ingestion',
      status: (source.test_status === 'success' ? 'succeeded' : source.test_status === 'failed' ? 'failed' : 'completed') as JobStatus,
      start_time: source.created_at,
      end_time: source.updated_at ?? null,
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

  // Per-node card-summary entries for trainer/tuner nodes. Trainer jobs
  // run via Celery so the engine's per-node `metadata.summary` doesn't
  // reach `executionResult.node_results` on the FE — this endpoint
  // surfaces the same one-liner from the latest completed run group
  // per node. For canvases with a parallel terminal (one training node
  // fed by N branches), the array contains one entry per branch so the
  // card can render Path A / Path B / … on separate lines.
  getNodeSummaries: async (limit: number = 200): Promise<Record<string, NodeSummaryEntry[]>> => {
    const response = await apiClient.get<Record<string, NodeSummaryEntry[]>>(
      `/pipeline/jobs/node-summaries`,
      { params: { limit } },
    );
    return response.data;
  },

  getBestTuningJob: async (modelType: string): Promise<JobInfo | null> => {
    const response = await apiClient.get<JobInfo | null>(`/pipeline/jobs/tuning/best/${modelType}`);
    return response.data;
  },

  getTuningHistory: async (modelType: string): Promise<JobInfo[]> => {
    const response = await apiClient.get<JobInfo[]>(`/pipeline/jobs/tuning/history/${modelType}`);
    return response.data;
  },

  promoteJob: async (jobId: string): Promise<void> => {
    await apiClient.post(`/pipeline/jobs/${jobId}/promote`);
  },

  unpromoteJob: async (jobId: string): Promise<void> => {
    await apiClient.delete(`/pipeline/jobs/${jobId}/promote`);
  }
};
