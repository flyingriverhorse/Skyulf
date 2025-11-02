// @ts-nocheck
export type FeatureNodeParameterSource = {
  type: string;
  endpoint: string;
  value_key?: string;
};

export type FeatureNodeParameterOption = {
  value: string;
  label: string;
  description?: string;
  metadata?: Record<string, any>;
};

export type FeatureNodeParameter = {
  name: string;
  label: string;
  description?: string;
  type: 'number' | 'multi_select' | 'text' | 'boolean' | 'select' | 'textarea';
  default?: any;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
  options?: FeatureNodeParameterOption[];
  source?: FeatureNodeParameterSource;
};

export type FeatureNodeCatalogEntry = {
  type: string;
  label: string;
  description: string;
  inputs: string[];
  outputs: string[];
  category?: string;
  tags?: string[];
  parameters?: FeatureNodeParameter[];
  default_config?: Record<string, any>;
};

export type DropColumnCandidate = {
  name: string;
  reason: string;
  missing_percentage?: number;
  priority?: string;
  signals?: string[];
  tags?: string[];
};

export type DropColumnRecommendationFilter = {
  id: string;
  label: string;
  description?: string;
  count: number;
};

export type DropColumnRecommendationsResponse = {
  dataset_source_id: string;
  suggested_threshold?: number;
  candidates: DropColumnCandidate[];
  generated_at?: string;
  available_filters?: DropColumnRecommendationFilter[];
  all_columns?: string[];
  column_missing_map?: Record<string, number>;
};

export type FetchDropColumnRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type LabelEncodingSuggestionStatus =
  | 'recommended'
  | 'high_cardinality'
  | 'identifier'
  | 'free_text'
  | 'single_category'
  | 'too_many_categories';

export type LabelEncodingColumnSuggestion = {
  column: string;
  status: LabelEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  score?: number | null;
  selectable: boolean;
};

export type LabelEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  auto_detect_default: boolean;
  high_cardinality_columns: string[];
  notes?: string[];
  columns: LabelEncodingColumnSuggestion[];
};

export type FetchLabelEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type OrdinalEncodingSuggestionStatus = LabelEncodingSuggestionStatus;

export type OrdinalEncodingColumnSuggestion = {
  column: string;
  status: OrdinalEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  score?: number | null;
  selectable: boolean;
  recommended_handle_unknown: boolean;
};

export type OrdinalEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  auto_detect_default: boolean;
  enable_unknown_default: boolean;
  high_cardinality_columns: string[];
  notes?: string[];
  columns: OrdinalEncodingColumnSuggestion[];
};

export type FetchOrdinalEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type TargetEncodingSuggestionStatus = LabelEncodingSuggestionStatus;

export type TargetEncodingColumnSuggestion = {
  column: string;
  status: TargetEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  score?: number | null;
  selectable: boolean;
  recommended_use_global_fallback: boolean;
};

export type TargetEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  auto_detect_default: boolean;
  enable_global_fallback_default: boolean;
  high_cardinality_columns: string[];
  notes?: string[];
  columns: TargetEncodingColumnSuggestion[];
};

export type FetchTargetEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type HashEncodingSuggestionStatus = LabelEncodingSuggestionStatus;

export type HashEncodingColumnSuggestion = {
  column: string;
  status: HashEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  score?: number | null;
  selectable: boolean;
  recommended_bucket_count: number;
};

export type HashEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  auto_detect_default: boolean;
  suggested_bucket_default: number;
  high_cardinality_columns: string[];
  notes?: string[];
  columns: HashEncodingColumnSuggestion[];
};

export type FetchHashEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type OneHotEncodingSuggestionStatus =
  | 'recommended'
  | 'high_cardinality'
  | 'identifier'
  | 'free_text'
  | 'single_category'
  | 'too_many_categories';

export type OneHotEncodingColumnSuggestion = {
  column: string;
  status: OneHotEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  estimated_dummy_columns: number;
  score?: number | null;
  selectable: boolean;
  recommended_drop_first: boolean;
};

export type OneHotEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  cautioned_count: number;
  high_cardinality_columns: string[];
  notes?: string[];
  columns: OneHotEncodingColumnSuggestion[];
};

export type FetchOneHotEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type DummyEncodingSuggestionStatus =
  | 'recommended'
  | 'high_cardinality'
  | 'identifier'
  | 'free_text'
  | 'single_category'
  | 'too_many_categories';

export type DummyEncodingColumnSuggestion = {
  column: string;
  status: DummyEncodingSuggestionStatus;
  reason: string;
  dtype?: string | null;
  unique_count?: number | null;
  unique_percentage?: number | null;
  missing_percentage?: number | null;
  text_category?: string | null;
  sample_values: string[];
  estimated_dummy_columns: number;
  score?: number | null;
  selectable: boolean;
  recommended_drop_first: boolean;
};

export type DummyEncodingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at?: string | null;
  sample_size: number;
  total_text_columns: number;
  recommended_count: number;
  cautioned_count: number;
  high_cardinality_columns: string[];
  notes?: string[];
  auto_detect_default: boolean;
  columns: DummyEncodingColumnSuggestion[];
};

export type FetchDummyEncodingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export type SkewnessMethodStatus = {
  status: 'ready' | 'unsupported';
  reason?: string;
};

export type SkewnessColumnDistribution = {
  bin_edges: number[];
  counts: number[];
  sample_size: number;
  missing_count: number;
  minimum?: number | null;
  maximum?: number | null;
  mean?: number | null;
  median?: number | null;
  stddev?: number | null;
};

export type SkewnessTransformationSelection = {
  column: string;
  method: string;
};

export type SkewnessMethodDetail = {
  key: string;
  label: string;
  description?: string;
  direction_bias?: 'left' | 'right' | 'either';
  requires_positive: boolean;
  supports_zero: boolean;
  supports_negative: boolean;
};

export type SkewnessColumnRecommendation = {
  column: string;
  skewness: number;
  direction: 'left' | 'right';
  magnitude: 'moderate' | 'substantial' | 'extreme';
  summary: string;
  recommended_methods: string[];
  method_status: Record<string, SkewnessMethodStatus>;
  applied_method?: string | null;
  distribution_before?: SkewnessColumnDistribution | null;
  distribution_after?: SkewnessColumnDistribution | null;
  /** @deprecated retained for backward compatibility */
  distribution?: SkewnessColumnDistribution | null;
};

export type SkewnessRecommendationsResponse = {
  dataset_source_id: string;
  generated_at: string;
  sample_size: number;
  skewness_threshold: number;
  methods: SkewnessMethodDetail[];
  columns: SkewnessColumnRecommendation[];
};

export type ScalingMethodName = 'standard' | 'minmax' | 'maxabs' | 'robust';

export type ScalingColumnStats = {
  valid_count: number;
  mean?: number | null;
  median?: number | null;
  stddev?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  iqr?: number | null;
  skewness?: number | null;
  outlier_ratio?: number | null;
};

export type ScalingMethodDetail = {
  key: ScalingMethodName;
  label: string;
  description?: string;
  handles_negative: boolean;
  handles_zero: boolean;
  handles_outliers: boolean;
  strengths: string[];
  cautions: string[];
};

export type ScalingColumnRecommendation = {
  column: string;
  dtype?: string | null;
  recommended_method: ScalingMethodName;
  confidence: 'high' | 'medium' | 'low';
  reasons: string[];
  fallback_methods: ScalingMethodName[];
  stats: ScalingColumnStats;
  has_missing: boolean;
};

export type ScalingRecommendationsResponse = {
  dataset_source_id: string;
  generated_at: string;
  sample_size: number;
  methods: ScalingMethodDetail[];
  columns: ScalingColumnRecommendation[];
};

export type PolynomialGeneratedFeature = {
  column: string;
  degree: number;
  terms: string[];
  expression: string;
  raw_feature?: string | null;
};

export type PolynomialFeaturesNodeSignal = {
  node_id?: string | null;
  configured_columns: string[];
  evaluated_columns: string[];
  applied_columns: string[];
  auto_detect: boolean;
  degree: number;
  include_bias: boolean;
  interaction_only: boolean;
  include_input_features: boolean;
  output_prefix: string;
  generated_columns: string[];
  generated_features: PolynomialGeneratedFeature[];
  skipped_columns: string[];
  filled_columns: Record<string, number>;
  feature_count: number;
  transform_mode?: string | null;
  notes: string[];
};

export type FeatureSelectionFeatureSummary = {
  column: string;
  selected: boolean;
  score?: number | null;
  p_value?: number | null;
  rank?: number | null;
  importance?: number | null;
  note?: string | null;
};

export type FeatureSelectionNodeSignal = {
  node_id?: string | null;
  method?: string | null;
  score_func?: string | null;
  mode?: string | null;
  estimator?: string | null;
  problem_type?: string | null;
  target_column?: string | null;
  configured_columns: string[];
  evaluated_columns: string[];
  selected_columns: string[];
  dropped_columns: string[];
  feature_summaries: FeatureSelectionFeatureSummary[];
  drop_unselected: boolean;
  auto_detect: boolean;
  k?: number | null;
  percentile?: number | null;
  alpha?: number | null;
  threshold?: number | null;
  transform_mode?: string | null;
  notes: string[];
};

export type BinnedColumnBin = {
  label: string;
  count: number;
  percentage: number;
  is_missing?: boolean;
};

export type BinnedColumnDistribution = {
  column: string;
  source_column?: string | null;
  total_rows: number;
  non_missing_rows: number;
  missing_rows: number;
  distinct_bins: number;
  top_label?: string | null;
  top_count?: number | null;
  top_percentage?: number | null;
  bins: BinnedColumnBin[];
};

export type BinnedDistributionResponse = {
  dataset_source_id: string;
  generated_at: string;
  sample_size: number;
  columns: BinnedColumnDistribution[];
};

export type QuickProfileDatasetMetrics = {
  row_count: number;
  column_count: number;
  missing_cells: number;
  missing_percentage: number;
  duplicate_rows: number;
  unique_rows: number;
};

export type QuickProfileNumericSummary = {
  mean: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  percentile_25: number | null;
  percentile_50: number | null;
  percentile_75: number | null;
};

export type QuickProfileValueCount = {
  value: any;
  count: number;
  percentage: number;
};

export type QuickProfileColumnSummary = {
  name: string;
  dtype: string | null;
  semantic_type?: string | null;
  missing_count: number;
  missing_percentage: number;
  distinct_count: number | null;
  sample_values: any[];
  numeric_summary?: QuickProfileNumericSummary | null;
  top_values: QuickProfileValueCount[];
};

export type QuickProfileCorrelation = {
  column_a: string;
  column_b: string;
  coefficient: number;
};

export type QuickProfileResponse = {
  dataset_source_id: string;
  generated_at: string;
  sample_size: number;
  rows_analyzed: number;
  columns_analyzed: number;
  metrics: QuickProfileDatasetMetrics;
  columns: QuickProfileColumnSummary[];
  correlations: QuickProfileCorrelation[];
  warnings: string[];
};

export type OutlierMethod = 'z_score' | 'iqr' | 'winsorize' | 'manual';

export type ManualOutlierBounds = {
  lower?: number | null;
  upper?: number | null;
};

export type OutlierDiagnosticsConfig = {
  method: OutlierMethod;
  columns: string[];
  z_threshold: number;
  iqr_multiplier: number;
  lower_percentile: number;
  upper_percentile: number;
  manual_bounds: Record<string, ManualOutlierBounds>;
};

export type OutlierDiagnosticsColumnDetail = {
  column: string;
  status: 'evaluated' | 'skipped';
  method: OutlierMethod;
  threshold: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  removed_rows: number;
  total_rows: number;
  removed_percentage: number;
  reason?: string | null;
  mean?: number | null;
  stddev?: number | null;
  median?: number | null;
  q1?: number | null;
  q3?: number | null;
  iqr?: number | null;
  lower_percentile?: number | null;
  upper_percentile?: number | null;
  manual_lower?: number | null;
  manual_upper?: number | null;
};

export type OutlierDiagnosticsResponse = {
  dataset_source_id: string;
  method: OutlierMethod;
  columns_evaluated: string[];
  passthrough_columns: string[];
  sample_size: number;
  total_rows: number;
  removed_rows: number;
  removed_percentage: number;
  kept_rows: number;
  kept_percentage: number;
  before_preview: Record<string, any>[];
  after_preview: Record<string, any>[];
  current_preview: Record<string, any>[];
  flagged_preview: Record<string, any>[];
  details: OutlierDiagnosticsColumnDetail[];
  config: OutlierDiagnosticsConfig;
  has_upstream_removal: boolean;
  used_full_dataset: boolean;
};

export type OutlierMethodName = 'iqr' | 'zscore' | 'elliptic_envelope' | 'winsorize' | 'manual';

export type OutlierMethodDetail = {
  key: OutlierMethodName;
  label: string;
  description?: string | null;
  action: 'remove' | 'cap' | 'manual';
  notes: string[];
  default_parameters: Record<string, number>;
  parameter_help: Record<string, string>;
};

export type OutlierMethodSummary = {
  method: OutlierMethodName;
  action: 'remove' | 'cap' | 'manual';
  lower_bound?: number | null;
  upper_bound?: number | null;
  affected_rows: number;
  affected_ratio: number;
  notes: string[];
};

export type OutlierColumnStats = {
  valid_count: number;
  mean?: number | null;
  median?: number | null;
  stddev?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  q1?: number | null;
  q3?: number | null;
  iqr?: number | null;
};

export type OutlierColumnInsight = {
  column: string;
  dtype?: string | null;
  stats: OutlierColumnStats;
  method_summaries: OutlierMethodSummary[];
  recommended_method?: OutlierMethodName | null;
  recommended_reason?: string | null;
  has_missing: boolean;
};

export type OutlierRecommendationsResponse = {
  dataset_source_id: string;
  sample_size: number;
  default_method: OutlierMethodName;
  methods: OutlierMethodDetail[];
  columns: OutlierColumnInsight[];
};

export type OutlierAppliedColumnSignal = {
  column: string;
  method: OutlierMethodName;
  action: 'remove' | 'cap';
  affected_rows: number;
  total_rows: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  notes: string[];
};

export type OutlierNodeSignal = {
  node_id?: string | null;
  configured_columns: string[];
  evaluated_columns: string[];
  default_method: OutlierMethodName;
  column_methods: Record<string, OutlierMethodName>;
  auto_detect: boolean;
  skipped_columns: string[];
  applied_columns: OutlierAppliedColumnSignal[];
  removed_rows: number;
  clipped_columns: string[];
};

export type TransformerSplitActivityAction =
  | 'fit_transform'
  | 'transform'
  | 'not_applied'
  | 'not_available';

export type TransformerSplitActivitySignal = {
  split: 'train' | 'test' | 'validation' | 'other' | 'unknown';
  action: TransformerSplitActivityAction;
  rows?: number | null;
  updated_at?: string | null;
  label?: string | null;
};

export type TransformerAuditEntrySignal = {
  source_node_id?: string | null;
  source_node_label?: string | null;
  transformer_name: string;
  column_name?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  split_activity: TransformerSplitActivitySignal[];
  metadata?: Record<string, any> | null;
  storage_key?: string | null;
};

export type TransformerAuditNodeSignal = {
  node_id?: string | null;
  pipeline_id?: string | null;
  transformers: TransformerAuditEntrySignal[];
  total_transformers: number;
  notes: string[];
};

export type FeatureTargetSplitNodeSignal = {
  node_id?: string | null;
  target_column?: string | null;
  configured_feature_columns: string[];
  feature_columns: string[];
  auto_included_columns: string[];
  missing_feature_columns: string[];
  excluded_columns: string[];
  feature_missing_counts: Record<string, number>;
  target_dtype?: string | null;
  target_missing_count: number;
  target_missing_percentage?: number | null;
  preview_row_count: number;
  total_rows: number;
  warnings: string[];
  notes: string[];
};

export type FeatureMathOperationStatus = 'applied' | 'skipped' | 'failed';

export type FeatureMathOperationResult = {
  operation_id: string;
  operation_type: string;
  method?: string | null;
  output_columns: string[];
  status: FeatureMathOperationStatus;
  message?: string | null;
};

export type FeatureMathNodeSignal = {
  node_id?: string | null;
  total_operations: number;
  applied_operations: number;
  skipped_operations: number;
  failed_operations: number;
  operations: FeatureMathOperationResult[];
  warnings: string[];
  generated_at?: string | null;
};

export type DatasetSourceSummary = {
  id: number;
  source_id: string;
  name?: string | null;
  description?: string | null;
  created_at?: string | null;
  is_owned?: boolean;
};

export type FeatureGraph = {
  nodes: any[];
  edges: any[];
};

export type FeaturePipelinePayload = {
  name?: string;
  description?: string;
  graph: FeatureGraph;
  metadata?: Record<string, any> | null;
};

export type FeaturePipelineResponse = FeaturePipelinePayload & {
  id: number;
  dataset_source_id: string;
  is_active: boolean;
  created_at?: string | null;
  updated_at?: string | null;
};

export type PipelinePreviewMetrics = {
  row_count: number;
  column_count: number;
  duplicate_rows: number;
  missing_cells: number;
  preview_rows: number;
  total_rows: number;
  requested_sample_size: number;
};

export type PipelinePreviewColumnStat = {
  name: string;
  dtype?: string | null;
  missing_count: number;
  missing_percentage: number;
  distinct_count?: number | null;
  mean?: number | null;
  median?: number | null;
  mode?: string | number | null;
};

export type PipelinePreviewRowStat = {
  index: number;
  missing_percentage: number;
};

export type PipelinePreviewRequest = {
  dataset_source_id: string;
  graph: FeatureGraph;
  target_node_id?: string | null;
  sample_size?: number;
  include_signals?: boolean;
  include_preview_rows?: boolean;
};

export type PipelinePreviewColumnSchema = {
  name: string;
  pandas_dtype?: string | null;
  logical_family:
    | 'numeric'
    | 'integer'
    | 'categorical'
    | 'string'
    | 'datetime'
    | 'boolean'
    | 'unknown';
  nullable: boolean;
};

export type PipelinePreviewSchema = {
  signature?: string | null;
  columns: PipelinePreviewColumnSchema[];
};

export type FullExecutionSignal = {
  status: 'succeeded' | 'deferred' | 'skipped' | 'failed';
  reason?: string | null;
  total_rows?: number | null;
  processed_rows?: number | null;
  applied_steps?: string[];
  warnings?: string[];
  dataset_source_id?: string | null;
  job_id?: string | null;
  job_status?: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | null;
  last_updated?: string | null;
  poll_after_seconds?: number | null;
  eta_seconds?: number | null;
};

export type PipelinePreviewSignals = {
  feature_math?: FeatureMathNodeSignal[];
  feature_target_split?: FeatureTargetSplitNodeSignal[];
  outlier_removal?: OutlierNodeSignal[];
  transformer_audit?: TransformerAuditNodeSignal[];
  polynomial_features?: PolynomialFeaturesNodeSignal[];
  feature_selection?: FeatureSelectionNodeSignal[];
  full_execution?: FullExecutionSignal | null;
  [key: string]: any;
};

export type PipelinePreviewResponse = {
  node_id?: string | null;
  columns: string[];
  sample_rows: Record<string, any>[];
  metrics: PipelinePreviewMetrics;
  column_stats: PipelinePreviewColumnStat[];
  applied_steps: string[];
  row_missing_stats: PipelinePreviewRowStat[];
  schema?: PipelinePreviewSchema | null;
  modeling_signals?: any;
  signals?: PipelinePreviewSignals | null;
};

export type PipelinePreviewRowsResponse = {
  columns: string[];
  rows: Record<string, any>[];
  offset: number;
  limit: number;
  returned_rows: number;
  total_rows?: number | null;
  next_offset?: number | null;
  has_more: boolean;
  sampling_mode?: string | null;
  sampling_adjustments: string[];
  large_dataset: boolean;
};

export type FetchPipelinePreviewRowsOptions = {
  offset?: number;
  limit?: number;
  mode?: 'head' | 'tail' | 'window';
};

export type TrainingJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export type TrainingJobResponse = {
  id: string;
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  user_id?: number | null;
  status: TrainingJobStatus;
  version: number;
  model_type: string;
  hyperparameters?: Record<string, any> | null;
  metadata?: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  artifact_uri?: string | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  graph: FeatureGraph;
  target_node_id?: string | null;
};

export type TrainingJobBatchResponse = {
  jobs: TrainingJobResponse[];
  total?: number;
};

export type TrainingJobSummary = {
  id: string;
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  status: TrainingJobStatus;
  version: number;
  model_type: string;
  metadata?: Record<string, any> | null;
  problem_type?: string | null;
  metrics?: Record<string, any> | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type TrainingJobListResponse = {
  jobs: TrainingJobSummary[];
  total: number;
};

export type ModelEvaluationConfusionMatrix = {
  labels: string[];
  matrix: number[][];
  normalized?: number[][] | null;
  totals: number[];
  accuracy?: number | null;
};

export type ModelEvaluationRocCurve = {
  label: string;
  fpr: number[];
  tpr: number[];
  thresholds: number[];
  auc?: number | null;
};

export type ModelEvaluationPrecisionRecallCurve = {
  label: string;
  recall: number[];
  precision: number[];
  thresholds: number[];
  average_precision?: number | null;
};

export type ModelEvaluationResidualHistogram = {
  bin_edges: number[];
  counts: number[];
};

export type ModelEvaluationResidualPoint = {
  actual: number;
  predicted: number;
};

export type ModelEvaluationResiduals = {
  histogram: ModelEvaluationResidualHistogram;
  scatter: ModelEvaluationResidualPoint[];
  summary: Record<string, number>;
};

export type ModelEvaluationSplitPayload = {
  split: string;
  row_count: number;
  metrics: Record<string, number | string | boolean | null>;
  confusion_matrix?: ModelEvaluationConfusionMatrix | null;
  roc_curves: ModelEvaluationRocCurve[];
  pr_curves: ModelEvaluationPrecisionRecallCurve[];
  residuals?: ModelEvaluationResiduals | null;
  notes: string[];
};

export type ModelEvaluationReport = {
  job_id: string;
  pipeline_id?: string | null;
  node_id?: string | null;
  generated_at: string;
  problem_type: 'classification' | 'regression';
  target_column?: string | null;
  splits: Record<string, ModelEvaluationSplitPayload>;
};

export type ModelEvaluationRequest = {
  splits?: string[] | null;
  include_confusion?: boolean;
  include_curves?: boolean;
  include_residuals?: boolean;
  max_curve_points?: number | null;
  max_scatter_points?: number | null;
};

export type TrainingJobCreatePayload = {
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  model_type?: string;
  model_types?: string[];
  hyperparameters?: Record<string, any> | null;
  metadata?: Record<string, any> | null;
  job_metadata?: Record<string, any> | null;
  run_training?: boolean;
  graph: FeatureGraph;
  target_node_id?: string | null;
};

export type HyperparameterTuningJobStatus = TrainingJobStatus;

export type HyperparameterTuningJobResponse = {
  id: string;
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  user_id?: number | null;
  status: HyperparameterTuningJobStatus;
  run_number: number;
  model_type: string;
  search_strategy: 'grid' | 'random';
  search_space?: Record<string, any> | null;
  baseline_hyperparameters?: Record<string, any> | null;
  n_iterations?: number | null;
  scoring?: string | null;
  random_state?: number | null;
  cross_validation?: Record<string, any> | null;
  metadata?: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  results?: Array<Record<string, any>> | null;
  best_params?: Record<string, any> | null;
  best_score?: number | null;
  artifact_uri?: string | null;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  graph: FeatureGraph;
  target_node_id?: string | null;
};

export type HyperparameterTuningJobBatchResponse = {
  jobs: HyperparameterTuningJobResponse[];
  total?: number;
};

export type HyperparameterTuningJobSummary = {
  id: string;
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  status: HyperparameterTuningJobStatus;
  run_number: number;
  model_type: string;
  search_strategy: 'grid' | 'random';
  metadata?: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  results?: Array<Record<string, any>> | null;
  best_params?: Record<string, any> | null;
  best_score?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type HyperparameterTuningJobListResponse = {
  jobs: HyperparameterTuningJobSummary[];
  total: number;
};

export type HyperparameterTuningJobCreatePayload = {
  dataset_source_id: string;
  pipeline_id: string;
  node_id: string;
  model_type?: string;
  model_types?: string[];
  search_strategy: 'grid' | 'random';
  search_space: Record<string, any>;
  baseline_hyperparameters?: Record<string, any> | null;
  n_iterations?: number | null;
  scoring?: string | null;
  random_state?: number | null;
  cross_validation?: Record<string, any> | null;
  metadata?: Record<string, any> | null;
  job_metadata?: Record<string, any> | null;
  run_tuning?: boolean;
  graph: FeatureGraph;
  target_node_id?: string | null;
};

export async function fetchNodeCatalog(): Promise<FeatureNodeCatalogEntry[]> {
  const response = await fetch('/ml-workflow/api/node-catalog');

  if (!response.ok) {
    throw new Error('Failed to load node catalog');
  }

  return response.json();
}

export async function fetchDropColumnRecommendations(
  sourceId: string,
  options: FetchDropColumnRecommendationsOptions = {}
): Promise<DropColumnRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/drop-columns', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load column drop recommendations');
  }

  return response.json();
}

export async function fetchLabelEncodingRecommendations(
  sourceId: string,
  options: FetchLabelEncodingRecommendationsOptions = {}
): Promise<LabelEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/label-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load label encoding recommendations');
  }

  return response.json();
}

export async function fetchOrdinalEncodingRecommendations(
  sourceId: string,
  options: FetchOrdinalEncodingRecommendationsOptions = {}
): Promise<OrdinalEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/ordinal-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load ordinal encoding suggestions');
  }

  return response.json();
}

export async function fetchTargetEncodingRecommendations(
  sourceId: string,
  options: FetchTargetEncodingRecommendationsOptions = {}
): Promise<TargetEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/target-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load target encoding recommendations');
  }

  return response.json();
}

export async function fetchHashEncodingRecommendations(
  sourceId: string,
  options: FetchHashEncodingRecommendationsOptions = {}
): Promise<HashEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/hash-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load hash encoding recommendations');
  }

  return response.json();
}

export async function fetchOneHotEncodingRecommendations(
  sourceId: string,
  options: FetchOneHotEncodingRecommendationsOptions = {}
): Promise<OneHotEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/one-hot-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load one-hot encoding recommendations');
  }

  return response.json();
}

export async function fetchDummyEncodingRecommendations(
  sourceId: string,
  options: FetchDummyEncodingRecommendationsOptions = {}
): Promise<DummyEncodingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/dummy-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load dummy encoding recommendations');
  }

  return response.json();
}

export type FetchSkewnessRecommendationsOptions = {
  sampleSize?: number;
  transformations?: SkewnessTransformationSelection[] | Record<string, string> | null;
  graph?: FeatureGraph | null;
  targetNodeId?: string | null;
};

export async function fetchSkewnessRecommendations(
  sourceId: string,
  options: FetchSkewnessRecommendationsOptions = {}
): Promise<SkewnessRecommendationsResponse> {
  const { sampleSize = 500, transformations, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (Array.isArray(transformations)) {
    if (transformations.length > 0) {
      body.transformations = JSON.stringify(transformations);
    }
  } else if (transformations && typeof transformations === 'object') {
    const keys = Object.keys(transformations);
    if (keys.length > 0) {
      body.transformations = JSON.stringify(transformations);
    }
  }

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/skewness', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load skewness recommendations');
  }

  return response.json();
}

export type FetchScalingRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | null;
  targetNodeId?: string | null;
};

export async function fetchScalingRecommendations(
  sourceId: string,
  options: FetchScalingRecommendationsOptions = {}
): Promise<ScalingRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/scaling', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load scaling recommendations');
  }

  return response.json();
}

export type FetchOutlierRecommendationsOptions = {
  sampleSize?: number;
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export async function fetchOutlierRecommendations(
  sourceId: string,
  options: FetchOutlierRecommendationsOptions = {}
): Promise<OutlierRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    typeof sampleSize === 'number' && Number.isFinite(sampleSize)
      ? Math.min(5000, Math.max(50, Math.round(sampleSize)))
      : 500;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: normalizedSampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/outliers', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load outlier recommendations');
  }

  return response.json();
}

export type FetchBinnedDistributionOptions = {
  sampleSize?: number | 'all';
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export async function fetchBinnedDistribution(
  sourceId: string,
  options: FetchBinnedDistributionOptions = {}
): Promise<BinnedDistributionResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    sampleSize === 'all'
      ? 0
      : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
        ? Math.max(0, Math.round(sampleSize))
        : 500;

  const params = new URLSearchParams({
    dataset_source_id: sourceId,
    sample_size: String(normalizedSampleSize),
  });

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    params.set('graph', JSON.stringify({ nodes: graph.nodes, edges: graph.edges }));
  }

  if (targetNodeId) {
    params.set('target_node_id', targetNodeId);
  }

  const response = await fetch(`/ml-workflow/api/analytics/binned-distribution?${params.toString()}`);

  if (!response.ok) {
    throw new Error('Failed to load binned column distributions');
  }

  return response.json();
}

export type FetchQuickProfileOptions = {
  sampleSize?: number | 'all';
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export async function fetchQuickProfile(
  sourceId: string,
  options: FetchQuickProfileOptions = {}
): Promise<QuickProfileResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    sampleSize === 'all'
      ? 0
      : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
        ? Math.max(0, Math.round(sampleSize))
        : 500;

  const params = new URLSearchParams({
    dataset_source_id: sourceId,
    sample_size: String(normalizedSampleSize),
  });

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    params.set('graph', JSON.stringify({ nodes: graph.nodes, edges: graph.edges }));
  }

  if (targetNodeId) {
    params.set('target_node_id', targetNodeId);
  }

  const response = await fetch(`/ml-workflow/api/analytics/quick-profile?${params.toString()}`);

  if (!response.ok) {
    let message = 'Failed to generate dataset profile';
    try {
      const errorPayload = await response.clone().json();
      const detail =
        (typeof errorPayload?.detail === 'string' && errorPayload.detail.trim())
          ? errorPayload.detail.trim()
          : (typeof errorPayload?.error === 'string' && errorPayload.error.trim())
            ? errorPayload.error.trim()
            : (typeof errorPayload?.message === 'string' && errorPayload.message.trim())
              ? errorPayload.message.trim()
              : null;
      if (detail) {
        message = detail;
      }
    } catch {
      try {
        const fallbackText = await response.text();
        const trimmed = fallbackText.trim();
        if (trimmed) {
          message = trimmed;
        }
      } catch {
        /* swallow - retain default message */
      }
    }
    throw new Error(message);
  }

  return response.json();
}

export type FetchOutlierDiagnosticsOptions = {
  config?: Partial<OutlierDiagnosticsConfig> | null;
  sampleSize?: number | 'all';
  graph?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

export async function fetchOutlierDiagnostics(
  sourceId: string,
  options: FetchOutlierDiagnosticsOptions = {}
): Promise<OutlierDiagnosticsResponse> {
  const { config, sampleSize, graph, targetNodeId } = options;

  const payload: Record<string, any> = {
    dataset_source_id: sourceId,
    sample_size:
      sampleSize === 'all'
        ? 0
        : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
          ? sampleSize
          : 300,
  };

  if (config && typeof config === 'object') {
    const clampPercentile = (value: any, fallback: number) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return fallback;
      }
      if (numeric < 0) {
        return 0;
      }
      if (numeric > 100) {
        return 100;
      }
      return numeric;
    };

    const sanitizeManualBounds = (raw: any): Record<string, ManualOutlierBounds> => {
      if (!raw || typeof raw !== 'object') {
        return {};
      }
      const result: Record<string, ManualOutlierBounds> = {};
      Object.entries(raw as Record<string, any>).forEach(([key, value]) => {
        const column = String(key ?? '').trim();
        if (!column) {
          return;
        }

        let lower: number | null = null;
        let upper: number | null = null;

        if (value && typeof value === 'object') {
          const lowerCandidate = (value as any).lower ?? (value as any).min ?? (value as any).minimum ?? null;
          const upperCandidate = (value as any).upper ?? (value as any).max ?? (value as any).maximum ?? null;

          const parsedLower = Number(lowerCandidate);
          if (Number.isFinite(parsedLower)) {
            lower = parsedLower;
          }

          const parsedUpper = Number(upperCandidate);
          if (Number.isFinite(parsedUpper)) {
            upper = parsedUpper;
          }
        }

        if (lower === null && upper === null) {
          return;
        }

        if (lower !== null && upper !== null && lower > upper) {
          const temp = lower;
          lower = upper;
          upper = temp;
        }

        result[column] = { lower, upper };
      });
      return result;
    };

    const lowerPercentile = clampPercentile((config as any).lower_percentile, 5);
    const upperPercentile = clampPercentile((config as any).upper_percentile, 95);
    payload.config = {
      ...config,
      method: config.method ?? 'z_score',
      columns: Array.isArray(config.columns) ? config.columns : [],
      z_threshold:
        typeof config.z_threshold === 'number' && Number.isFinite(config.z_threshold)
          ? config.z_threshold
          : 3,
      iqr_multiplier:
        typeof config.iqr_multiplier === 'number' && Number.isFinite(config.iqr_multiplier)
          ? config.iqr_multiplier
          : 1.5,
      lower_percentile: lowerPercentile,
      upper_percentile: upperPercentile <= lowerPercentile ? Math.min(lowerPercentile + 1, 100) : upperPercentile,
      manual_bounds: sanitizeManualBounds((config as any).manual_bounds),
    };
  }

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    payload.graph = {
      nodes: graph.nodes,
      edges: graph.edges,
    };
  }

  if (targetNodeId) {
    payload.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/analytics/outliers', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load outlier diagnostics');
  }

  return response.json();
}

export async function fetchDatasets(limit = 8): Promise<DatasetSourceSummary[]> {
  const response = await fetch(`/ml-workflow/api/datasets?limit=${encodeURIComponent(String(limit))}`);

  if (!response.ok) {
    throw new Error('Failed to load datasets');
  }

  return response.json();
}

export async function fetchPipeline(
  sourceId: string
): Promise<FeaturePipelineResponse | null> {
  const response = await fetch(`/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}`);

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error('Failed to load saved pipeline');
  }

  return response.json();
}

export async function savePipeline(
  sourceId: string,
  payload: FeaturePipelinePayload
): Promise<FeaturePipelineResponse> {
  const response = await fetch(`/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Failed to save pipeline');
  }

  return response.json();
}

export async function fetchPipelineHistory(
  sourceId: string,
  limit = 10
): Promise<FeaturePipelineResponse[]> {
  const response = await fetch(
  `/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}/history?limit=${encodeURIComponent(
      String(limit)
    )}`
  );

  if (response.status === 404) {
    return [];
  }

  if (!response.ok) {
    throw new Error('Failed to load pipeline history');
  }

  return response.json();
}

export async function fetchPipelinePreviewRows(
  datasetSourceId: string,
  options: FetchPipelinePreviewRowsOptions = {}
): Promise<PipelinePreviewRowsResponse> {
  const { offset, limit, mode } = options;

  const params = new URLSearchParams();

  if (typeof offset === 'number' && Number.isFinite(offset)) {
    params.set('offset', String(offset));
  }

  if (typeof limit === 'number' && Number.isFinite(limit)) {
    params.set('limit', String(limit));
  }

  if (mode) {
    params.set('mode', mode);
  }

  const query = params.toString();
  const url = `/ml-workflow/api/pipelines/${encodeURIComponent(datasetSourceId)}/preview/rows${
    query ? `?${query}` : ''
  }`;

  const response = await fetch(url);

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load preview rows');
  }

  return response.json();
}

export async function fetchPipelinePreview(
  payload: PipelinePreviewRequest
): Promise<PipelinePreviewResponse> {
  const response = await fetch('/ml-workflow/api/pipelines/preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load pipeline preview');
  }

  return response.json();
}

const sortGraphValue = (value: any): any => {
  if (Array.isArray(value)) {
    return value.map((item) => sortGraphValue(item));
  }
  if (value && typeof value === 'object') {
    const sortedKeys = Object.keys(value).sort();
    const result: Record<string, any> = {};
    sortedKeys.forEach((key) => {
      result[key] = sortGraphValue(value[key]);
    });
    return result;
  }
  return value;
};

const toHex = (buffer: ArrayBuffer): string => {
  return Array.from(new Uint8Array(buffer))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};

const sha256Hex = async (value: string): Promise<string> => {
  if (!globalThis.crypto || !globalThis.crypto.subtle) {
    throw new Error('Secure hashing is not available in this environment');
  }
  const encoder = new TextEncoder();
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoder.encode(value));
  return toHex(digest);
};

const sanitizeGraphForHash = (graph: FeatureGraph | null | undefined): FeatureGraph => {
  const rawNodes = graph && Array.isArray(graph.nodes) ? graph.nodes : [];
  const rawEdges = graph && Array.isArray(graph.edges) ? graph.edges : [];

  const safeNodes = rawNodes
    .map((node: any) => {
      let config = null;
      if (node && node.data && typeof node.data === 'object') {
        const candidate = node.data.config;
        if (candidate !== undefined) {
          try {
            config = JSON.parse(JSON.stringify(candidate));
          } catch (error) {
            config = candidate;
          }
        }
      }
      return {
        id: node && Object.prototype.hasOwnProperty.call(node, 'id') ? node.id ?? null : null,
        type: node && Object.prototype.hasOwnProperty.call(node, 'type') ? node.type ?? null : null,
        catalogType:
          node && node.data && Object.prototype.hasOwnProperty.call(node.data, 'catalogType')
            ? node.data.catalogType ?? null
            : null,
        config: config,
      };
    })
    .sort((a, b) => {
      const first = typeof a.id === 'string' ? a.id : String(a.id ?? '');
      const second = typeof b.id === 'string' ? b.id : String(b.id ?? '');
      return first.localeCompare(second);
    });

  const safeEdges = rawEdges
    .map((edge: any) => ({
      source: edge && Object.prototype.hasOwnProperty.call(edge, 'source') ? edge.source ?? null : null,
      target: edge && Object.prototype.hasOwnProperty.call(edge, 'target') ? edge.target ?? null : null,
      sourceHandle:
        edge && Object.prototype.hasOwnProperty.call(edge, 'sourceHandle') ? edge.sourceHandle ?? null : null,
      targetHandle:
        edge && Object.prototype.hasOwnProperty.call(edge, 'targetHandle') ? edge.targetHandle ?? null : null,
    }))
    .sort((a, b) => {
      const sourceCompare = String(a.source ?? '').localeCompare(String(b.source ?? ''));
      if (sourceCompare !== 0) {
        return sourceCompare;
      }
      return String(a.target ?? '').localeCompare(String(b.target ?? ''));
    });

  return {
    nodes: safeNodes,
    edges: safeEdges,
  };
};

export async function generatePipelineId(
  datasetSourceId: string,
  graph: FeatureGraph | null | undefined
): Promise<string> {
  if (!datasetSourceId) {
    throw new Error('datasetSourceId is required to compute pipeline ID');
  }

  const safeGraph = sanitizeGraphForHash(graph);
  const graphJson = JSON.stringify(sortGraphValue(safeGraph));
  const hash = await sha256Hex(graphJson);
  return `${datasetSourceId}_${hash.slice(0, 8)}`;
}

export async function createTrainingJob(
  payload: TrainingJobCreatePayload
): Promise<TrainingJobBatchResponse> {
  const response = await fetch('/ml-workflow/api/training-jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to launch training jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to enqueue training job');
  }

  const data = await response.json();

  if (data && Array.isArray(data?.jobs)) {
    const jobs: TrainingJobResponse[] = data.jobs;
    const total = typeof data.total === 'number' ? data.total : jobs.length;
    return { jobs, total };
  }

  if (data && data.id) {
    return { jobs: [data as TrainingJobResponse], total: 1 };
  }

  return { jobs: [], total: 0 };
}

export async function fetchTrainingJob(jobId: string): Promise<TrainingJobResponse> {
  if (!jobId) {
    throw new Error('jobId is required');
  }

  const response = await fetch(`/ml-workflow/api/training-jobs/${encodeURIComponent(jobId)}`, {
    credentials: 'include',
  });

  if (response.status === 404) {
    throw new Error('Training job not found');
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view this training job.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load training job');
  }

  return response.json();
}

export type FetchTrainingJobsOptions = {
  datasetSourceId?: string;
  pipelineId?: string;
  nodeId?: string;
  limit?: number;
};

export async function fetchTrainingJobs(
  options: FetchTrainingJobsOptions = {}
): Promise<TrainingJobListResponse> {
  const params = new URLSearchParams();

  if (options.datasetSourceId) {
    params.set('dataset_source_id', options.datasetSourceId);
  }

  if (options.pipelineId) {
    params.set('pipeline_id', options.pipelineId);
  }

  if (options.nodeId) {
    params.set('node_id', options.nodeId);
  }

  if (typeof options.limit === 'number' && Number.isFinite(options.limit)) {
    params.set('limit', String(options.limit));
  }

  const query = params.toString();
  const response = await fetch(`/ml-workflow/api/training-jobs${query ? `?${query}` : ''}`, {
    credentials: 'include',
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view training jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load training jobs');
  }

  return response.json();
}

export async function evaluateTrainingJob(
  jobId: string,
  payload: ModelEvaluationRequest
): Promise<ModelEvaluationReport> {
  if (!jobId) {
    throw new Error('jobId is required for evaluation');
  }

  const response = await fetch(
    `/ml-workflow/api/training-jobs/${encodeURIComponent(jobId)}/evaluate`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(payload ?? {}),
    }
  );

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to evaluate this model.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to evaluate model');
  }

  return response.json();
}

export async function createHyperparameterTuningJob(
  payload: HyperparameterTuningJobCreatePayload
): Promise<HyperparameterTuningJobBatchResponse> {
  const response = await fetch('/ml-workflow/api/hyperparameter-tuning-jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to launch tuning jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to enqueue tuning job');
  }

  const data = await response.json();

  if (data && Array.isArray(data?.jobs)) {
    const jobs: HyperparameterTuningJobResponse[] = data.jobs;
    const total = typeof data.total === 'number' ? data.total : jobs.length;
    return { jobs, total };
  }

  if (data && data.id) {
    return { jobs: [data as HyperparameterTuningJobResponse], total: 1 };
  }

  return { jobs: [], total: 0 };
}

export async function fetchHyperparameterTuningJob(
  jobId: string
): Promise<HyperparameterTuningJobResponse> {
  if (!jobId) {
    throw new Error('jobId is required');
  }

  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning-jobs/${encodeURIComponent(jobId)}`,
    {
      credentials: 'include',
    }
  );

  if (response.status === 404) {
    throw new Error('Tuning job not found');
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view this tuning job.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load tuning job');
  }

  return response.json();
}

export type FetchHyperparameterTuningJobsOptions = {
  datasetSourceId?: string;
  pipelineId?: string;
  nodeId?: string;
  limit?: number;
};

export async function fetchHyperparameterTuningJobs(
  options: FetchHyperparameterTuningJobsOptions = {}
): Promise<HyperparameterTuningJobListResponse> {
  const params = new URLSearchParams();

  if (options.datasetSourceId) {
    params.set('dataset_source_id', options.datasetSourceId);
  }

  if (options.pipelineId) {
    params.set('pipeline_id', options.pipelineId);
  }

  if (options.nodeId) {
    params.set('node_id', options.nodeId);
  }

  if (typeof options.limit === 'number' && Number.isFinite(options.limit)) {
    params.set('limit', String(options.limit));
  }

  const query = params.toString();
  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning-jobs${query ? `?${query}` : ''}`,
    {
      credentials: 'include',
    }
  );

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view tuning jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load tuning jobs');
  }

  return response.json();
}

export async function triggerFullDatasetExecution(
  payload: PipelinePreviewRequest
): Promise<PipelinePreviewResponse> {
  // Same as preview but with sample_size = 0 to force full dataset
  const fullDatasetPayload = {
    ...payload,
    sample_size: 0, // Force full dataset execution
  };

  const response = await fetch('/ml-workflow/api/pipelines/preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(fullDatasetPayload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to trigger full dataset execution');
  }

  return response.json();
}

export type ModelHyperparameterField = {
  name: string;
  label: string;
  type: 'number' | 'select' | 'boolean' | 'text';
  default: any;
  description?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string; label: string }>;
  nullable?: boolean;
};

export type ModelHyperparametersResponse = {
  model_type: string;
  fields: ModelHyperparameterField[];
  defaults: Record<string, any>;
};

export async function fetchModelHyperparameters(
  modelType: string
): Promise<ModelHyperparametersResponse> {
  const response = await fetch(`/ml-workflow/api/model-hyperparameters/${encodeURIComponent(modelType)}`);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model type '${modelType}' not found`);
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load model hyperparameters');
  }

  return response.json();
}

export type BestHyperparametersResponse = {
  available: boolean;
  model_type: string;
  message?: string;
  job_id?: string;
  pipeline_id?: string;
  node_id?: string;
  run_number?: number;
  best_params?: Record<string, any>;
  best_score?: number;
  scoring?: string;
  finished_at?: string;
  search_strategy?: string;
  n_iterations?: number;
};

export type FetchBestHyperparametersOptions = {
  pipelineId?: string;
  datasetSourceId?: string;
};

export async function fetchBestHyperparameters(
  modelType: string,
  options: FetchBestHyperparametersOptions = {}
): Promise<BestHyperparametersResponse> {
  const params = new URLSearchParams();
  
  if (options.pipelineId) {
    params.set('pipeline_id', options.pipelineId);
  }
  if (options.datasetSourceId) {
    params.set('dataset_source_id', options.datasetSourceId);
  }

  const query = params.toString();
  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning/best-params/${encodeURIComponent(modelType)}${query ? `?${query}` : ''}`,
    {
      credentials: 'include',
    }
  );

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model type '${modelType}' not found`);
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch best hyperparameters');
  }

  return response.json();
}

export async function fetchFullExecutionStatus(
  datasetSourceId: string,
  jobId: string
): Promise<FullExecutionSignal> {
  const response = await fetch(
    `/ml-workflow/api/pipelines/${encodeURIComponent(datasetSourceId)}/full-execution/${encodeURIComponent(jobId)}`
  );

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch full execution status');
  }

  return response.json();
}
