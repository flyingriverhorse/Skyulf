export interface ScalingConfig {
  columns: string[];
  method: 'standard' | 'minmax' | 'maxabs' | 'robust';

  // Standard Scaler
  with_mean?: boolean;
  with_std?: boolean;

  // MinMax Scaler
  feature_range_min?: number | null;
  feature_range_max?: number | null;

  // Robust Scaler
  quantile_range_min?: number | null;
  quantile_range_max?: number | null;
  with_centering?: boolean;
  with_scaling?: boolean;
}

export interface ScalingSettingsProps {
  config: ScalingConfig;
  onChange: (config: ScalingConfig) => void;
  nodeId?: string;
}
