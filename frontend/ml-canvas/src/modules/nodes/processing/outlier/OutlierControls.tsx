import type { OutlierConfig, OutlierSettingsProps } from './types';

function OutlierMethod({ config, onChange }: OutlierSettingsProps) {
  return (
    <div>
      <span className="block text-sm font-medium mb-1">Method</span>
      <select
        aria-label="Method"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.method}
        onChange={(e) => onChange({ ...config, method: e.target.value as OutlierConfig['method'] })}
      >
        <option value="iqr">IQR (Interquartile Range)</option>
        <option value="zscore">Z-Score (Standard Deviation)</option>
        <option value="winsorize">Winsorize (Clip Values)</option>
        <option value="elliptic_envelope">Elliptic Envelope (Multivariate)</option>
      </select>
      <p className="text-[10px] text-muted-foreground mt-1">
        {config.method === 'iqr' && 'Removes rows with values outside Q1/Q3 ± multiplier * IQR.'}
        {config.method === 'zscore' && 'Removes rows with values more than N standard deviations from mean.'}
        {config.method === 'winsorize' && 'Clips values to specified percentiles instead of removing rows.'}
        {config.method === 'elliptic_envelope' && 'Fits a robust covariance estimate to detect outliers.'}
      </p>
    </div>
  );
}

function IqrOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Multiplier</span>
      <input
        aria-label="Multiplier"
        type="number"
        step="0.1"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.multiplier ?? 1.5}
        onChange={(e) => onChange({ ...config, multiplier: Number.parseFloat(e.target.value) })}
      />
      <p className="text-xs text-muted-foreground">Usually 1.5 for outliers, 3.0 for extreme outliers.</p>
    </div>
  );
}

function ZscoreOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Threshold (Sigma)</span>
      <input
        aria-label="Threshold (Sigma)"
        type="number"
        step="0.1"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.threshold ?? 3.0}
        onChange={(e) => onChange({ ...config, threshold: Number.parseFloat(e.target.value) })}
      />
      <p className="text-xs text-muted-foreground">Number of standard deviations to tolerate.</p>
    </div>
  );
}

function WinsorizeOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Percentiles</span>
      <div className="flex gap-2 items-center">
        <div className="flex-1">
          <span className="text-[10px] text-muted-foreground">Lower</span>
          <input
            aria-label="Lower Percentile"
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.lower_percentile ?? 5.0}
            onChange={(e) => onChange({ ...config, lower_percentile: Number.parseFloat(e.target.value) })}
          />
        </div>
        <div className="flex-1">
          <span className="text-[10px] text-muted-foreground">Upper</span>
          <input
            aria-label="Upper Percentile"
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.upper_percentile ?? 95.0}
            onChange={(e) => onChange({ ...config, upper_percentile: Number.parseFloat(e.target.value) })}
          />
        </div>
      </div>
    </div>
  );
}

function EllipticEnvelopeOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Contamination</span>
      <input
        aria-label="Contamination"
        type="number"
        step="0.01"
        min="0"
        max="0.5"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.contamination ?? 0.01}
        onChange={(e) => onChange({ ...config, contamination: Number.parseFloat(e.target.value) })}
      />
      <p className="text-xs text-muted-foreground">Expected proportion of outliers in the dataset.</p>
    </div>
  );
}

export function OutlierControls({ config, onChange }: OutlierSettingsProps) {
  return (
    <>
      <OutlierMethod config={config} onChange={onChange} />
      {config.method === 'iqr' && <IqrOptions config={config} onChange={onChange} />}
      {config.method === 'zscore' && <ZscoreOptions config={config} onChange={onChange} />}
      {config.method === 'winsorize' && <WinsorizeOptions config={config} onChange={onChange} />}
      {config.method === 'elliptic_envelope' && <EllipticEnvelopeOptions config={config} onChange={onChange} />}
    </>
  );
}
