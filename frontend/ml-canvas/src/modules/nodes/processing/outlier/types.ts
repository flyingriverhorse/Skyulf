export interface OutlierConfig {
  method: 'iqr' | 'zscore' | 'winsorize' | 'elliptic_envelope';
  columns: string[];

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
