export interface ManualColumnBounds {
  lower?: number | null;
  upper?: number | null;
}

export interface OutlierConfig {
  method: 'iqr' | 'zscore' | 'winsorize' | 'elliptic_envelope' | 'manual_bounds' | 'clip';
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

  // Elliptic Envelope
  contamination?: number;
}

export interface OutlierSettingsProps {
  config: OutlierConfig;
  onChange: (config: OutlierConfig) => void;
  nodeId?: string;
}
