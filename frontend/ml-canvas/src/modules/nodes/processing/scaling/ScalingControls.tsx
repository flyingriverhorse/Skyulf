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
      </select>
      <p className="text-[10px] text-muted-foreground mt-1">
        {config.method === 'standard' && 'Centers data around 0 with unit variance.'}
        {config.method === 'minmax' && 'Scales data to a fixed range [0, 1].'}
        {config.method === 'maxabs' && 'Scales data by its maximum absolute value.'}
        {config.method === 'robust' && 'Scales data using statistics that are robust to outliers.'}
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

function MinmaxOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <div className="space-y-2 border-t pt-2">
      <span className="block text-sm font-medium">Feature Range</span>
      <div className="flex gap-2 items-center">
        <input
          aria-label="Feature Range Minimum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Min (0)"
          value={config.feature_range_min ?? 0}
          onChange={(e) => onChange({ ...config, feature_range_min: Number.parseFloat(e.target.value) })}
        />
        <span className="text-muted-foreground">-</span>
        <input
          aria-label="Feature Range Maximum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Max (1)"
          value={config.feature_range_max ?? 1}
          onChange={(e) => onChange({ ...config, feature_range_max: Number.parseFloat(e.target.value) })}
        />
      </div>
    </div>
  );
}

function RobustOptions({ config, onChange }: ScalingSettingsProps) {
  return (
    <div className="space-y-2 border-t pt-2">
      <span className="block text-sm font-medium">Quantile Range</span>
      <div className="flex gap-2 items-center">
        <input
          aria-label="Quantile Range Minimum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Min (25.0)"
          value={config.quantile_range_min ?? 25.0}
          onChange={(e) => onChange({ ...config, quantile_range_min: Number.parseFloat(e.target.value) })}
        />
        <span className="text-muted-foreground">-</span>
        <input
          aria-label="Quantile Range Maximum"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Max (75.0)"
          value={config.quantile_range_max ?? 75.0}
          onChange={(e) => onChange({ ...config, quantile_range_max: Number.parseFloat(e.target.value) })}
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
      </div>
    </div>
  );
}

export function ScalingControls({ config, onChange }: ScalingSettingsProps) {
  return (
    <>
      <ScalingMethod config={config} onChange={onChange} />
      {config.method === 'standard' && <StandardOptions config={config} onChange={onChange} />}
      {config.method === 'minmax' && <MinmaxOptions config={config} onChange={onChange} />}
      {config.method === 'robust' && <RobustOptions config={config} onChange={onChange} />}
    </>
  );
}
