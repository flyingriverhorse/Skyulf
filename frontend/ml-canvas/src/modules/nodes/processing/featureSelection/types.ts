export interface FeatureSelectionConfig {
  method:
  | 'variance_threshold'
  | 'correlation_threshold'
  | 'select_k_best'
  | 'select_percentile'
  | 'select_fpr'
  | 'select_fdr'
  | 'select_fwe'
  | 'generic_univariate_select'
  | 'select_from_model'
  | 'rfe';

  // Common
  target_column?: string | undefined;
  datasetId?: string | undefined;
  problem_type?: 'auto' | 'classification' | 'regression' | undefined;

  // Method Specific
  threshold?: number | string | undefined; // Variance, Correlation, SelectFromModel (can be "median")
  correlation_method?: 'pearson' | 'kendall' | 'spearman' | undefined;
  k?: number | undefined; // SelectKBest, RFE
  percentile?: number | undefined; // SelectPercentile
  alpha?: number | undefined; // FPR, FDR, FWE
  score_func?: string | undefined; // Univariate methods
  mode?: 'k_best' | 'percentile' | 'fpr' | 'fdr' | 'fwe' | undefined; // Generic
  param?: number | undefined; // Generic
  estimator?: 'RandomForest' | 'LogisticRegression' | 'LinearRegression' | 'auto' | undefined; // Model based
  max_features?: number | undefined; // SelectFromModel
  step?: number | undefined; // RFE
  drop_columns?: boolean | undefined;
}


export interface SelectionConfigProps {
  config: FeatureSelectionConfig;
  onChange: (config: FeatureSelectionConfig) => void;
}
