import { ValidationField } from '../../../../components/shared/ValidationField';
import { numericDraft, numericInputValue } from '../../../../core/utils/numericValidation';
import type { OutlierConfig, OutlierSettingsProps } from './types';
import { ManualBoundsOptions } from './ManualBoundsOptions';

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
        <option value="elliptic_envelope">Elliptic Envelope (Per-column)</option>
        <option value="isolation_forest">Isolation Forest (Multivariate)</option>
        <option value="local_outlier_factor">Local Outlier Factor (Multivariate)</option>
        <option value="manual_bounds">Manual Bounds (Filter Rows)</option>
        <option value="clip">Clip to Fixed Bounds (Keep Rows)</option>
      </select>
      <p className="text-[10px] text-muted-foreground mt-1">
        {config.method === 'iqr' && 'Removes rows with values outside Q1/Q3 ± multiplier * IQR.'}
        {config.method === 'zscore' && 'Removes rows with values more than N standard deviations from mean.'}
        {config.method === 'winsorize' && 'Clips values to specified percentiles instead of removing rows.'}
        {config.method === 'elliptic_envelope' && 'Fits a robust covariance estimate to detect outliers.'}
        {config.method === 'isolation_forest' && 'Learns joint isolation trees across the selected features.'}
        {config.method === 'local_outlier_factor' && 'Learns local density across selected features. Saved scoring compares new rows with the training population.'}
        {config.method === 'manual_bounds' && 'Rows outside these bounds are removed. Values are not clipped.'}
        {config.method === 'clip' && 'Values outside these bounds are capped at the bound. No rows are removed.'}
      </p>
    </div>
  );
}

function IqrOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Multiplier</span>
      <ValidationField field="multiplier"><input
        aria-label="Multiplier"
        type="number"
        step="0.1"
        className="w-full p-2 border rounded bg-background text-sm"
        value={numericInputValue(config.multiplier, 1.5)}
        onChange={(e) => onChange({ ...config, multiplier: numericDraft(e.target.value) })}
      /></ValidationField>
      <p className="text-xs text-muted-foreground">Usually 1.5 for outliers, 3.0 for extreme outliers.</p>
    </div>
  );
}

function ZscoreOptions({ config, onChange }: OutlierSettingsProps) {
  return (
    <div className="space-y-2">
      <span className="block text-sm font-medium">Threshold (Sigma)</span>
      <ValidationField field="threshold"><input
        aria-label="Threshold (Sigma)"
        type="number"
        step="0.1"
        className="w-full p-2 border rounded bg-background text-sm"
        value={numericInputValue(config.threshold, 3.0)}
        onChange={(e) => onChange({ ...config, threshold: numericDraft(e.target.value) })}
      /></ValidationField>
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
          <ValidationField field="lower_percentile"><input
            aria-label="Lower Percentile"
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.lower_percentile ?? 5.0}
            onChange={(e) => onChange({ ...config, lower_percentile: Number.parseFloat(e.target.value) })}
          /></ValidationField>
        </div>
        <div className="flex-1">
          <span className="text-[10px] text-muted-foreground">Upper</span>
          <ValidationField field="upper_percentile"><input
            aria-label="Upper Percentile"
            type="number"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.upper_percentile ?? 95.0}
            onChange={(e) => onChange({ ...config, upper_percentile: Number.parseFloat(e.target.value) })}
          /></ValidationField>
        </div>
      </div>
      <ValidationField field="percentiles_by_column"><ColumnPercentiles config={config} onChange={onChange} /></ValidationField>
    </div>
  );
}

function ColumnPercentiles({ config, onChange }: OutlierSettingsProps) {
  const update = (column: string, pair?: { lower: number; upper: number }) => {
    const overrides = { ...config.percentiles_by_column };
    if (pair) overrides[column] = pair;
    else delete overrides[column];
    onChange({ ...config, percentiles_by_column: overrides });
  };
  return <div className="space-y-2">{config.columns.map(column => {
    const pair = config.percentiles_by_column?.[column];
    return <div key={column} className="space-y-1 border-t pt-2">
      <label className="flex gap-2 text-xs"><input type="checkbox" checked={!!pair}
        onChange={event => update(column, event.target.checked ? { lower: config.lower_percentile ?? 5, upper: config.upper_percentile ?? 95 } : undefined)} />
        Override percentiles for {column}
      </label>
      {pair && <div className="flex gap-2">{(['lower', 'upper'] as const).map(tail => <ValidationField key={tail} field={`percentiles_by_column.${column}.${tail}`}>
        <input aria-label={`${column} ${tail} percentile`} type="number" min="0" max="100" className="w-full p-2 border rounded" value={numericInputValue(pair[tail], tail === 'lower' ? 5 : 95)}
          onChange={event => update(column, { ...pair, [tail]: numericDraft(event.target.value) })} />
      </ValidationField>)}</div>}
    </div>;
  })}</div>;
}

function JointDetectorOptions({ config, onChange }: OutlierSettingsProps) {
  const forest = config.method === 'isolation_forest';
  const field = forest ? 'n_estimators' : 'n_neighbors';
  return <div className="space-y-2">
    <label className="block text-xs">Contamination
      <ValidationField field="contamination"><input aria-label="Contamination" type="number" min="0.001" max="0.5" step="0.01" className="w-full p-2 border rounded"
        value={numericInputValue(config.contamination, 0.01)} onChange={event => onChange({ ...config, contamination: numericDraft(event.target.value) })} /></ValidationField>
    </label>
    <label className="block text-xs">{forest ? 'Trees' : 'Neighbors'}
      <ValidationField field={field}><input aria-label={forest ? 'Trees' : 'Neighbors'} type="number" min="1" step="1" className="w-full p-2 border rounded"
        value={numericInputValue(config[field], forest ? 100 : 20)} onChange={event => onChange({ ...config, [field]: numericDraft(event.target.value) })} /></ValidationField>
    </label>
    {forest && <label className="block text-xs">Random seed
      <ValidationField field="random_state"><input aria-label="Random seed" type="number" min="0" step="1" className="w-full p-2 border rounded" value={numericInputValue(config.random_state, 42)}
        onChange={event => onChange({ ...config, random_state: numericDraft(event.target.value) })} /></ValidationField>
    </label>}
    <p className="text-xs text-muted-foreground">Nonfinite rows remain unscored. Impute first if later steps require complete values.</p>
  </div>;
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
      {(config.method === 'isolation_forest' || config.method === 'local_outlier_factor') && <JointDetectorOptions config={config} onChange={onChange} />}
      {(config.method === 'manual_bounds' || config.method === 'clip') && <ManualBoundsOptions config={config} onChange={onChange} />}
    </>
  );
}
