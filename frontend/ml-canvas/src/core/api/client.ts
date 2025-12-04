import axios from 'axios';

const API_BASE = '/ml-workflow/api';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic_type?: string;
  missing_count: number;
  missing_percentage: number;
}

export interface QuickProfileResponse {
  dataset_source_id: string;
  columns: ColumnProfile[];
}

export const fetchDatasetProfile = async (sourceId: string): Promise<QuickProfileResponse> => {
  const response = await apiClient.get<QuickProfileResponse>('/analytics/quick-profile', {
    params: { dataset_source_id: sourceId },
  });
  return response.data;
};

export interface PipelinePreviewRequest {
  dataset_source_id: string;
  graph: {
    nodes: any[];
    edges: any[];
  };
  sample_size?: number;
}

export const runPipelinePreview = async (payload: PipelinePreviewRequest) => {
  const response = await apiClient.post('/pipelines/preview', payload);
  return response.data;
};

export interface FeaturePipelineCreate {
  name: string;
  description?: string;
  graph: {
    nodes: any[];
    edges: any[];
  };
  metadata?: any;
}

export interface FeaturePipelineResponse extends FeaturePipelineCreate {
  id: number;
  dataset_source_id: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export const fetchPipeline = async (datasetSourceId: string): Promise<FeaturePipelineResponse | null> => {
  try {
    const response = await apiClient.get<FeaturePipelineResponse>(`/pipelines/${encodeURIComponent(datasetSourceId)}`);
    return response.data;
  } catch (error: any) {
    if (error.response && error.response.status === 404) {
      return null;
    }
    throw error;
  }
};

export const savePipeline = async (datasetSourceId: string, payload: FeaturePipelineCreate): Promise<FeaturePipelineResponse> => {
  const response = await apiClient.post<FeaturePipelineResponse>(`/pipelines/${encodeURIComponent(datasetSourceId)}`, payload);
  return response.data;
};

export interface TrainingJobCreate {
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  model_types: string[];
  hyperparameters?: any;
  metadata?: any;
  run_training?: boolean;
  graph: {
    nodes: any[];
    edges: any[];
  };
  target_node_id?: string;
}

export const submitTrainingJob = async (payload: TrainingJobCreate) => {
  const response = await apiClient.post('/training-jobs', payload);
  return response.data;
};

// --- V2 API Client & Types ---

const API_V2_BASE = '/api';

export const apiClientV2 = axios.create({
  baseURL: API_V2_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

export const runPipelinePreviewV2 = async (payload: PipelineConfigModel) => {
  const response = await apiClientV2.post('/pipeline/preview', payload);
  return response.data;
};
