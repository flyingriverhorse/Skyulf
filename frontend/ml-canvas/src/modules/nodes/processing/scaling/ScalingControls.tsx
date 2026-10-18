import { ValidationField } from '../../../../components/shared/ValidationField';
import type { ScalingConfig, ScalingSettingsProps } from './types';

function ScalingMethod({ config, onChange }: ScalingSettingsProps) {
  return (
    <div>
      <span className="block text-sm font-medium mb-1">Scaling Method</span>
      <select
        aria-label="Scaling Method"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.method}
        onChange={(e) => onChange({ ...config, method: e.target.value as ScalingConfig['method'] })}
      >
        <option value="standard">Standard Scaler (Z-Score)</option>
        <option value="minmax">MinMax Scaler (0-1)</option>
        <option value="maxabs">MaxAbs Scaler</option>
        <option value="robust">Robust Scaler (Outlier Safe)</option>
        <option value="quantile">Quantile Transformer</option>
        <option value="normalize">Row Normalizer</option>
      </select>
      <p className="text-[10px] text-muted-foreground mt-1">
        {config.method === 'standard' && 'Centers data around 0 with unit variance.'}
        {config.method === 'minmax' && 'Scales data to a fixed range [0, 1].'}
        {config.method === 'maxabs' && 'Scales data by its maximum absolute value.'}
        {config.method === 'robust' && 'Scales data using statistics that are robust to outliers.'}
        {config.method === 'quantile' && 'Uses the training distribution to map each feature to uniform or normal values.'}
        {config.method === 'normalize' && 'Scales each row across the selected features. Fill missing values first.'}
      </p>
    </div>
  );
}

function StandardOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <div className="space-y-2 border-t pt-2">
      <label className="flex items-center gap-2 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={config.with_mean ?? true}
          onChange={(e) => onChange({ ...config, with_mean: e.target.checked })}
          className="rounded border-gray-300 text-primary focus:ring-primary"
        />
        <span>Center Data (with_mean)</span>
      </label>
      <label className="flex items-center gap-2 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={config.with_std ?? true}
          onChange={(e) => onChange({ ...config, with_std: e.target.checked })}
          className="rounded border-gray-300 text-primary focus:ring-primary"
        />
        <span>Scale Variance (with_std)</span>
      </label>
    </div>
  );
}

/** Keep missing defaults distinct from explicitly emptied or nonfinite drafts. */
function rangeInput(value: number | null | undefined, fallback: number): number | string {
  if (value === undefined) return fallback;
  return value !== null && Number.isFinite(value) ? value : '';
}

function MinmaxOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <ValidationField field="feature_range" className="space-y-2 border-t pt-2">
      <span className="block text-sm font-medium">Feature Range</span>
      <div className="flex gap-2 items-center">
        <input
          aria-label="Feature Range Minimum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Min (0)"
          value={rangeInput(config.feature_range_min, 0)}
          onChange={(e) => onChange({ ...config, feature_range_min: Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : null })}
        />
        <span className="text-muted-foreground">-</span>
        <input
          aria-label="Feature Range Maximum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Max (1)"
          value={rangeInput(config.feature_range_max, 1)}
          onChange={(e) => onChange({ ...config, feature_range_max: Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : null })}
        />
      </div>
      <label className="flex items-center gap-2 text-sm cursor-pointer">
        <input type="checkbox" checked={config.clip ?? false}
          onChange={(e) => onChange({ ...config, clip: e.target.checked })} />
        <span>Clip values to feature range</span>
      </label>
    </ValidationField>
  );
}

function RobustOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <ValidationField field="quantile_range" className="space-y-2 border-t pt-2">
      <span className="block text-sm font-medium">Quantile Range</span>
      <div className="flex gap-2 items-center">
        <input
          aria-label="Quantile Range Minimum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Min (25.0)"
          value={rangeInput(config.quantile_range_min, 25.0)}
          onChange={(e) => onChange({ ...config, quantile_range_min: Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : null })}
        />
        <span className="text-muted-foreground">-</span>
        <input
          aria-label="Quantile Range Maximum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Max (75.0)"
          value={rangeInput(config.quantile_range_max, 75.0)}
          onChange={(e) => onChange({ ...config, quantile_range_max: Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : null })}
        />
      </div>
      <div className="space-y-2 mt-2">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={config.with_centering ?? true}
            onChange={(e) => onChange({ ...config, with_centering: e.target.checked })}
            className="rounded border-gray-300 text-primary focus:ring-primary"
          />
          <span>Center Data (Median)</span>
        </label>
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={config.with_scaling ?? true}
            onChange={(e) => onChange({ ...config, with_scaling: e.target.checked })}
            className="rounded border-gray-300 text-primary focus:ring-primary"
          />
          <span>Scale Data (IQR)</span>
        </label>
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input type="checkbox" checked={config.unit_variance ?? false}
            onChange={(e) => onChange({ ...config, unit_variance: e.target.checked })} />
          <span>Unit variance for normal data</span>
        </label>
      </div>
    </ValidationField>
  );
}

function QuantileOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <div className="space-y-2 border-t pt-2">
      <ValidationField field="output_distribution">
        <label className="block text-sm font-medium">
          Output distribution
          <select aria-label="Output distribution" className="w-full p-2 border rounded bg-background text-sm"
            value={config.output_distribution ?? 'uniform'}
            onChange={(e) => onChange({ ...config, output_distribution: e.target.value as NonNullable<ScalingConfig['output_distribution']> })}>
            <option value="uniform">Uniform (0 to 1)</option>
            <option value="normal">Normal</option>
          </select>
        </label>
      </ValidationField>
      <ValidationField field="n_quantiles">
        <label className="block text-sm font-medium">
          Quantile count
          <input aria-label="Quantile count" type="number" min={1} step={1}
            className="w-full p-2 border rounded bg-background text-sm"
            value={rangeInput(config.n_quantiles, 1000)}
            onChange={(e) => onChange({ ...config, n_quantiles: Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : null })} />
        </label>
        <p className="text-[10px] text-muted-foreground mt-1">The fitted count is capped at the number of training rows.</p>
      </ValidationField>
    </div>
  );
}

function NormalizerOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <ValidationField field="norm" className="space-y-2 border-t pt-2">
      <label className="block text-sm font-medium">
        Row norm
        <select aria-label="Row norm" className="w-full p-2 border rounded bg-background text-sm"
          value={config.norm ?? 'l2'}
          onChange={(e) => onChange({ ...config, norm: e.target.value as NonNullable<ScalingConfig['norm']> })}>
          <option value="l1">L1 (absolute sum)</option>
          <option value="l2">L2 (Euclidean length)</option>
          <option value="max">Maximum absolute value</option>
        </select>
      </label>
      <p className="text-[10px] text-muted-foreground mt-1">Rows containing only zeros remain zero.</p>
    </ValidationField>
  );
}

export function ScalingControls({ config, onChange }: ScalingSettingsProps) {
  return (
    <>
      <ScalingMethod config={config} onChange={onChange} />
      {config.method === 'standard' && <StandardOptions config={config} onChange={onChange} />}
      {config.method === 'minmax' && <MinmaxOptions config={config} onChange={onChange} />}
      {config.method === 'robust' && <RobustOptions config={config} onChange={onChange} />}
      {config.method === 'quantile' && <QuantileOptions config={config} onChange={onChange} />}
      {config.method === 'normalize' && <NormalizerOptions config={config} onChange={onChange} />}
    </>
  );
}
