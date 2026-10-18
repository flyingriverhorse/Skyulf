export interface ManualColumnBounds {
  lower?: number | null;
  upper?: number | null;
}

export interface OutlierConfig {
  method: 'iqr' | 'zscore' | 'winsorize' | 'elliptic_envelope' | 'manual_bounds' | 'clip' | 'isolation_forest' | 'local_outlier_factor';
  columns: string[];

  // Manual Bounds and Clip (same fixed bounds; one drops rows, one caps values)
  bounds?: Record<string, ManualColumnBounds>;

  // IQR
  multiplier?: number;

  // Z-Score
  threshold?: number;

  // Winsorize
  lower_percentile?: number;
  upper_percentile?: number;
  percentiles_by_column?: Record<string, { lower: number; upper: number }>;

  // Elliptic Envelope
  contamination?: number;
  n_estimators?: number;
  n_neighbors?: number;
  random_state?: number;
}

export interface OutlierSettingsProps {
  config: OutlierConfig;
  onChange: (config: OutlierConfig) => void;
  nodeId?: string;
}
