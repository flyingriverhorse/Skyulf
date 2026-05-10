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
  /** Wall-clock duration of this node's last successful run, in
   * seconds. Source: backend `NodeExecutionResult.execution_time`.
   * Used by the L4 perf overlay to color-code node cards. */
  execution_time?: number;
  /** Optional engine-side per-node free-form metadata. `summary` is a
   * short one-line human string (e.g. "7,000 / 1,500 / 1,500" for a
   * train/test split) that the canvas renders inside the node body when
   * present. Backend may add other keys here later. */
  metadata?: {
    summary?: string;
    [key: string]: unknown;
  };
}

export type PreviewDataRows = Array<Record<string, unknown>>;
export type PreviewData = PreviewDataRows | Record<string, PreviewDataRows>;
/** Per-branch preview keyed by branch label (e.g. "Path A · Random Forest"). */
export type BranchPreviews = Record<string, PreviewData>;
/** Per-branch node IDs that ran in that branch. */
export type BranchNodeIds = Record<string, string[]>;

/** Engine-emitted advisory for a downstream node whose inputs share an
 * ancestor (column-union + last-wins merge semantics applied), or whose
 * row-wise merge had to drop non-shared columns. */
export interface MergeWarning {
  node_id: string;
  kind: string;
  inputs?: string[];
  common_ancestors?: string[];
  /** Columns present in 2+ inputs (subject to last-wins overwrite). */
  overlap_columns?: string[];
  /** ID of the input whose values won on overlap (always the last one). */
  winner_input?: string;
  /** Row-wise merge: split label ("train" / "test" / "rows"). */
  part?: string;
  /** Row-wise merge: columns present in some inputs but not all. */
  dropped_columns?: string[];
  /** Row-wise merge: columns kept (intersection). */
  kept_columns?: string[];
  message: string;
}

export interface PreviewResponse {
  pipeline_id: string;
  status: string;
  node_results: Record<string, NodeExecutionResult>;
  preview_data: PreviewData | null;
  /** True row counts per split key in `preview_data` (rows themselves are
   *  capped at 50 for transport). For single-frame previews exposed under
   *  the synthetic `_total` key. */
  preview_totals?: Record<string, number> | null;
  /** Set when the pipeline ran multiple parallel branches. */
  branch_previews?: BranchPreviews | null;
  /** Per-branch true row counts (mirrors `branch_previews` keys). */
  branch_preview_totals?: Record<string, Record<string, number>> | null;
  /** Per-branch list of node IDs (used to filter the applied-steps pills). */
  branch_node_ids?: BranchNodeIds | null;
  /** Advisories surfaced when the engine had to merge sibling fan-in inputs. */
  merge_warnings?: MergeWarning[];
  /** Soft per-node warnings (e.g. TargetEncoder coerced a float multiclass
   *  target, OneHotEncoder saw a degenerate category). Captured by the
   *  backend `WarningCaptureHandler` during the run. */
  node_warnings?: NodeWarning[];
  recommendations: Recommendation[];
}

export interface NodeWarning {
  node_id: string | null;
  node_type: string | null;
  level: string;
  logger: string;
  message: string;
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

