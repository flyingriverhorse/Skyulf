import axios from 'axios';

// --- V2 API Client ---
const API_V2_BASE = '/api';

export const apiClientV2 = axios.create({
  baseURL: API_V2_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- V2 Interfaces ---
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

// --- V2 Functions ---

export const runPipelinePreviewV2 = async (payload: PipelineConfigModel): Promise<PreviewResponse> => {
  const response = await apiClientV2.post<PreviewResponse>('/pipeline/preview', payload);
  return response.data;
};

export const fetchDatasetProfile = async (datasetId: string): Promise<AnalysisProfile> => {
  const response = await apiClientV2.get<AnalysisProfile>(`/pipeline/datasets/${datasetId}/schema`);
  return response.data;
};

// --- Placeholder Functions (To be implemented in V2) ---

export interface SavedPipeline {
  id: string;
  graph: {
    nodes: any[];
    edges: any[];
  };
  name?: string;
  description?: string;
}

export const savePipeline = async (datasetId: string, payload: any): Promise<{ id: string }> => {
  console.warn('savePipeline is not yet implemented in V2', datasetId, payload);
  return { id: 'mock_id' };
};

export const fetchPipeline = async (datasetId: string): Promise<SavedPipeline | null> => {
  console.warn('fetchPipeline is not yet implemented in V2', datasetId);
  return null; 
};

export const submitTrainingJob = async (payload: any): Promise<{ job_id: string }> => {
  console.warn('submitTrainingJob is not yet implemented in V2', payload);
  return { job_id: 'mock_job_id' };
};

