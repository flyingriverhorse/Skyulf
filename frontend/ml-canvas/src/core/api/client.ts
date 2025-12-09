import axios from 'axios';

// --- API Client ---
const API_BASE = '/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- Interfaces ---
export interface NodeConfigModel {
  node_id: string;
  step_type: string;
  params: Record<string, any>;
  inputs: string[];
}

export interface PipelineConfigModel {
  pipeline_id: string;
  nodes: NodeConfigModel[];
  metadata?: Record<string, any>;
  target_node_id?: string;
  job_type?: string;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  column_type?: string;
  missing_count: number;
  missing_ratio: number;
  unique_count: number;
}

export interface AnalysisProfile {
  columns: Record<string, ColumnProfile>;
  row_count: number;
}

export interface Recommendation {
  rule_id: string;
  type: string;
  target_columns: string[];
  description: string;
  suggested_node_type: string;
  suggested_params: Record<string, any>;
  confidence: number;
  reasoning: string;
}

export interface PreviewResponse {
  pipeline_id: string;
  status: string;
  node_results: Record<string, any>;
  preview_data: any;
  recommendations: Recommendation[];
}

// --- Functions ---

export const runPipelinePreview = async (payload: PipelineConfigModel): Promise<PreviewResponse> => {
  const response = await apiClient.post<PreviewResponse>('/pipeline/preview', payload);
  return response.data;
};

export const fetchDatasetProfile = async (datasetId: string): Promise<AnalysisProfile> => {
  const response = await apiClient.get<AnalysisProfile>(`/pipeline/datasets/${datasetId}/schema`);
  return response.data;
};

// --- Placeholder Functions ---

export interface SavedPipeline {
  id?: string; // Optional because backend might not return it on load
  graph: {
    nodes: any[];
    edges: any[];
  };
  name?: string;
  description?: string;
}

export const savePipeline = async (datasetId: string, payload: any): Promise<{ id: string }> => {
  const response = await apiClient.post<{ id: string }>(`/pipeline/save/${datasetId}`, payload);
  return response.data;
};

export const fetchPipeline = async (datasetId: string): Promise<SavedPipeline | null> => {
  const response = await apiClient.get<SavedPipeline | null>(`/pipeline/load/${datasetId}`);
  // Handle 204 No Content or null response
  if (!response.data) return null;
  return response.data;
};

export const submitTrainingJob = async (payload: PipelineConfigModel): Promise<{ job_id: string }> => {
  const response = await apiClient.post<{ job_id: string }>('/pipeline/run', payload);
  return response.data;
};

