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
  params: Record<string, unknown>;
  inputs: string[];
}

export interface PipelineConfigModel {
  pipeline_id: string;
  nodes: NodeConfigModel[];
  metadata?: Record<string, unknown>;
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
  min_value?: number;
  max_value?: number;
  mean_value?: number;
  std_value?: number;
}

export interface AnalysisProfile {
  columns: Record<string, ColumnProfile>;
  row_count: number;
  column_count?: number;
  duplicate_row_count?: number;
}

export interface Recommendation {
  rule_id: string;
  type: string;
  target_columns: string[];
  description: string;
  suggested_node_type: string;
  suggested_params: Record<string, unknown>;
  confidence: number;
  reasoning: string;
}

export interface NodeExecutionResult {
  status?: string;
  error?: string;
  metrics?: Record<string, unknown>;
  output?: unknown;
}

export type PreviewDataRows = Array<Record<string, unknown>>;
export type PreviewData = PreviewDataRows | Record<string, PreviewDataRows>;
/** Per-branch preview keyed by branch label (e.g. "Path A · Random Forest"). */
export type BranchPreviews = Record<string, PreviewData>;
/** Per-branch node IDs that ran in that branch. */
export type BranchNodeIds = Record<string, string[]>;

/** Engine-emitted advisory for a downstream node whose inputs share an
 * ancestor (column-union + last-wins merge semantics applied). */
export interface MergeWarning {
  node_id: string;
  kind: string;
  inputs: string[];
  common_ancestors: string[];
  /** Columns present in 2+ inputs (subject to last-wins overwrite). */
  overlap_columns?: string[];
  /** ID of the input whose values won on overlap (always the last one). */
  winner_input?: string;
  message: string;
}

export interface PreviewResponse {
  pipeline_id: string;
  status: string;
  node_results: Record<string, NodeExecutionResult>;
  preview_data: PreviewData | null;
  /** Set when the pipeline ran multiple parallel branches. */
  branch_previews?: BranchPreviews | null;
  /** Per-branch list of node IDs (used to filter the applied-steps pills). */
  branch_node_ids?: BranchNodeIds | null;
  /** Advisories surfaced when the engine had to merge sibling fan-in inputs. */
  merge_warnings?: MergeWarning[];
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
    nodes: unknown[];
    edges: unknown[];
  };
  name?: string;
  description?: string;
}

export const savePipeline = async (datasetId: string, payload: unknown): Promise<{ id: string }> => {
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

