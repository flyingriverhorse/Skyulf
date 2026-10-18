export interface ScalingConfig {
  columns: string[];
  method: 'standard' | 'minmax' | 'maxabs' | 'robust' | 'quantile' | 'normalize';

  // Standard Scaler
  with_mean?: boolean;
  with_std?: boolean;

  // MinMax Scaler
  feature_range_min?: number | null;
  feature_range_max?: number | null;
  clip?: boolean;

  // Robust Scaler
  quantile_range_min?: number | null;
  quantile_range_max?: number | null;
  with_centering?: boolean;
  with_scaling?: boolean;
  unit_variance?: boolean;

  // Quantile Transformer and row Normalizer
  n_quantiles?: number | null;
  output_distribution?: 'uniform' | 'normal';
  norm?: 'l1' | 'l2' | 'max';
}

export interface ScalingSettingsProps {
  config: ScalingConfig;
  onChange: (config: ScalingConfig) => void;
  nodeId?: string;
}
