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

export type BinningStrategyName = 'equal_width' | 'equal_frequency' | 'kbins';

export type BinningColumnStats = {
  valid_count: number;
  missing_count: number;
  distinct_count?: number | null;
  minimum?: number | null;
  maximum?: number | null;
  mean?: number | null;
  median?: number | null;
  stddev?: number | null;
  skewness?: number | null;
  has_negative: boolean;
  has_zero: boolean;
  has_positive: boolean;
};

export type BinningColumnRecommendation = {
  column: string;
  dtype?: string | null;
  recommended_strategy: BinningStrategyName;
  recommended_bins: number;
  confidence: 'high' | 'medium' | 'low';
  reasons: string[];
  notes: string[];
  stats: BinningColumnStats;
};

export type BinningExcludedColumn = {
  column: string;
  reason: string;
  dtype?: string | null;
};

export type BinningRecommendationsResponse = {
  dataset_source_id: string;
  generated_at: string;
  sample_size: number;
  columns: BinningColumnRecommendation[];
  excluded_columns: BinningExcludedColumn[];
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
  pending_nodes?: string[] | null;
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
  error_message?: string | null;
  progress?: number | null;
  current_step?: string | null;
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
  search_strategy: string;
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
  search_strategy: string;
  metadata?: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  results?: Array<Record<string, any>> | null;
  best_params?: Record<string, any> | null;
  best_score?: number | null;
  error_message?: string | null;
  progress?: number | null;
  current_step?: string | null;
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
  search_strategy: string;
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

export type FetchHyperparameterTuningJobsOptions = {
  datasetSourceId?: string;
  pipelineId?: string;
  nodeId?: string;
  limit?: number;
};

export type FetchBestHyperparametersOptions = {
  pipelineId?: string;
  datasetSourceId?: string;
};

export type ModelHyperparameterFieldOption = {
  value: string | number | boolean;
  label: string;
  description?: string;
  disabled?: boolean;
  metadata?: Record<string, any>;
};

export type ModelHyperparameterFieldOptionGroup = {
  label: string;
  options: ModelHyperparameterFieldOption[];
};

export type ModelHyperparameterField = {
  name: string;
  label: string;
  description?: string;
  type:
    | 'number'
    | 'integer'
    | 'float'
    | 'text'
    | 'textarea'
    | 'select'
    | 'multi_select'
    | 'json'
    | 'boolean'
    | 'object'
    | 'array'
    | string;
  required?: boolean;
  default?: any;
  placeholder?: string;
  help?: string;
  section?: string;
  group?: string;
  advanced?: boolean;
  warning?: string;
  depends_on?: string[];
  min?: number;
  max?: number;
  step?: number;
  precision?: number;
  options?: ModelHyperparameterFieldOption[];
  option_groups?: ModelHyperparameterFieldOptionGroup[];
  children?: ModelHyperparameterField[];
  metadata?: Record<string, any>;
  ui?: Record<string, any>;
  [key: string]: any;
};

export type ModelHyperparametersResponse = {
  model_type: string;
  problem_type?: string | null;
  defaults: Record<string, any>;
  fields: ModelHyperparameterField[];
  metadata?: Record<string, any> | null;
  version?: string | null;
};

export type BestHyperparametersResponse = {
  available: boolean;
  model_type: string;
  message?: string;
  job_id?: string;
  pipeline_id?: string | null;
  node_id?: string | null;
  run_number?: number;
  best_params?: Record<string, any>;
  best_score?: number | null;
  scoring?: string;
  finished_at?: string | null;
  search_strategy?: string;
  n_iterations?: number;
};

export type FetchTrainingJobsOptions = {
  datasetSourceId?: string;
  pipelineId?: string;
  nodeId?: string;
  limit?: number;
  offset?: number;
};
