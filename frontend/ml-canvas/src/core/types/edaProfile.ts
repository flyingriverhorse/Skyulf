/**
 * TypeScript types mirroring the EDA Pydantic schemas in
 * `skyulf-core/skyulf/profiling/schemas.py`.
 *
 * Use these in place of `any` when consuming `EDAService.analyze` /
 * `getReport` responses so TS catches shape drift at compile time.
 */

export type ColumnDtype = 'Numeric' | 'Categorical' | 'Boolean' | 'DateTime' | 'Text';

export interface HistogramBin {
  start: number;
  end: number;
  count: number;
}

export interface NumericStats {
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  variance?: number | null;
  min?: number | null;
  max?: number | null;
  q25?: number | null;
  q75?: number | null;
  skewness?: number | null;
  kurtosis?: number | null;
  zeros_count?: number | null;
  negatives_count?: number | null;
  normality_test?: Record<string, unknown> | null;
}

export interface CategoricalTopK {
  value: string;
  count: number;
}

export interface CategoricalStats {
  unique_count: number;
  top_k: CategoricalTopK[];
  rare_labels_count: number;
}

export interface TextStats {
  avg_length?: number | null;
  min_length?: number | null;
  max_length?: number | null;
  common_words: Array<{ word: string; count: number } | Record<string, unknown>>;
  sentiment_distribution?: Record<string, number> | null;
}

export interface DateStats {
  min_date?: string | null;
  max_date?: string | null;
  range_days?: number | null;
}

export interface NormalityTestResult {
  test: string;
  statistic: number;
  p_value: number;
  is_normal: boolean;
}

export interface OutlierPoint {
  index: number;
  value: number;
  score?: number;
}

export interface OutlierAnalysis {
  method: 'IsolationForest' | 'IQR' | string;
  total_outliers: number;
  outlier_percentage: number;
  top_outliers: OutlierPoint[];
  plot_data?: Array<Record<string, unknown>> | null;
}

export interface ColumnProfile {
  name: string;
  dtype: ColumnDtype | string;
  missing_count: number;
  missing_percentage: number;
  numeric_stats?: NumericStats | null;
  categorical_stats?: CategoricalStats | null;
  date_stats?: DateStats | null;
  text_stats?: TextStats | null;
  histogram?: HistogramBin[] | null;
  normality_test?: NormalityTestResult | null;
  is_constant?: boolean;
  is_unique?: boolean;
}

export interface EDAAlert {
  severity: 'info' | 'warning' | 'error' | string;
  message: string;
  column?: string;
}

export interface EDAProfile {
  row_count: number;
  column_count: number;
  columns: Record<string, ColumnProfile>;
  outliers?: OutlierAnalysis | null;
  correlations?: Record<string, Record<string, number>> | null;
  alerts?: EDAAlert[];
  sample_data?: Array<Record<string, unknown>>;
  /** Catch-all for future fields not yet typed (PCA, clustering, geo, etc.). */
  [extra: string]: unknown;
}
