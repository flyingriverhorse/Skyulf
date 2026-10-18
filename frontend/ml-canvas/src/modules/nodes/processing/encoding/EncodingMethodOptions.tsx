import { ValidationField } from '../../../../components/shared/ValidationField';
import { parseIntSafe } from '../../../../core/utils/numberInput';
import type { AnalysisProfile } from '../../../../core/api/client';
import type { EncodingConfig, EncodingSettingsProps } from './types';
import { OnehotCategories } from './OnehotCategories';

interface EncodingMethodOptionsProps extends EncodingSettingsProps {
  schema: AnalysisProfile | undefined;
  categoricalColumns: string[];
}

function OnehotOptions({ config, onChange }: EncodingSettingsProps) {
  const fixed = config.categories !== undefined;
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={config.drop_first || false}
          onChange={(e) => onChange({ ...config, drop_first: e.target.checked })}
          className="rounded border-gray-300"
        />
        Drop First Category
      </label>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Handle Unknown</span>
        <select
          aria-label="Handle Unknown"
          className="w-full p-1 text-sm border rounded"
          value={config.handle_unknown || 'ignore'}
          onChange={(e) => onChange({ ...config, handle_unknown: e.target.value as EncodingConfig['handle_unknown'] })}
          title="How to handle categories seen in test data but not in training data."
        >
          <option value="ignore">Ignore (Zeros)</option>
          <option value="error">Raise Error</option>
        </select>
        <p className="text-[10px] text-muted-foreground">
          &quot;Ignore&quot; produces all-zeros for unknown categories.
        </p>
      </div>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Max Categories</span>
        <input
          aria-label="Max Categories"
          type="number"
          min="2"
          max="200"
          className="w-full p-1 text-sm border rounded"
          value={config.max_categories === null ? '' : (config.max_categories ?? 20)}
          disabled={fixed}
          onChange={(e) => onChange({ ...config, max_categories: parseIntSafe(e.target.value, config.max_categories ?? undefined) })}
          title="Caps the number of one-hot columns per feature."
        />
        <p className="text-[10px] text-muted-foreground">Caps columns per feature (default 20).</p>
      </div>
      <label className="flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={config.include_missing || false}
          disabled={fixed}
          onChange={(e) => onChange({ ...config, include_missing: e.target.checked })}
          className="rounded border-gray-300"
        />
        Include missing as category
      </label>
      <OnehotCategories config={config} onChange={onChange} />
    </div>
  );
}

function DummyOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={config.drop_first || false}
          onChange={(e) => onChange({ ...config, drop_first: e.target.checked })}
          className="rounded border-gray-300"
        />
        Drop First Category
      </label>
      <p className="text-[10px] text-muted-foreground ml-5">
        Recommended: avoids the dummy variable trap (multicollinearity).
      </p>
    </div>
  );
}

function TargetOptions({ config, onChange, schema, categoricalColumns }: EncodingMethodOptionsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <span className="block text-sm font-medium">Target Column</span>
      <ValidationField field="target_column">
        <select
          aria-label="Target Column"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.target_column || ''}
          onChange={(e) => onChange({ ...config, target_column: e.target.value })}
        >
          <option value="">Select a target column...</option>
          {categoricalColumns.map(name => {
            const col = schema?.columns[name];
            return (
              <option key={name} value={name}>{name}{col ? ` (${col.dtype})` : ''}</option>
            );
          })}
        </select>
      </ValidationField>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Smoothing</span>
        <input
          aria-label="Smoothing"
          type="text"
          className="w-full p-1 text-sm border rounded"
          value={config.smooth ?? 'auto'}
          onChange={(e) => {
            const v = e.target.value;
            onChange({ ...config, smooth: v === 'auto' ? 'auto' : (Number.isNaN(Number(v)) ? 'auto' : Number(v)) });
          }}
          placeholder="auto or number"
          title="Smoothing strength. 'auto' uses sklearn's default."
        />
        <p className="text-[10px] text-muted-foreground">Higher = shrinks more toward global mean (default: auto).</p>
      </div>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Target Type</span>
        <select
          aria-label="Target Type"
          className="w-full p-1 text-sm border rounded"
          value={config.target_type || 'auto'}
          onChange={(e) => onChange({ ...config, target_type: e.target.value as EncodingConfig['target_type'] })}
        >
          <option value="auto">Auto-detect</option>
          <option value="continuous">Continuous (regression)</option>
          <option value="binary">Binary classification</option>
          <option value="multiclass">Multiclass classification</option>
        </select>
      </div>
      <TargetCrossFitOptions config={config} onChange={onChange} />
    </div>
  );
}

function TargetCrossFitOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2">
      <ValidationField field="cv">
      <label className="block text-xs">Maximum CV folds
        <input aria-label="Maximum CV folds" type="number" min="2" step="1" className="w-full p-1 border rounded"
          value={config.cv ?? 5} onChange={(e) => onChange({ ...config, cv: Number(e.target.value) })} />
      </label>
      </ValidationField>
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={config.shuffle ?? true}
          onChange={(e) => onChange({ ...config, shuffle: e.target.checked })} />
        Shuffle folds
      </label>
      <ValidationField field="random_state">
      <label className="block text-xs">Random seed
        <input aria-label="Random seed" type="number" min="0" max="4294967295" step="1" className="w-full p-1 border rounded"
          value={config.random_state === null ? '' : (config.random_state ?? 42)} disabled={config.shuffle === false}
          onChange={(e) => onChange({ ...config, random_state: e.target.value === '' ? null : Number(e.target.value) })} />
      </label>
      </ValidationField>
      <p className="text-[10px] text-muted-foreground">Folds are capped by available rows and class counts. Disabling shuffle does not provide temporal or grouped CV.</p>
    </div>
  );
}

function FrequencyOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border">
      <label className="block text-xs">Statistic
        <select aria-label="Statistic" className="w-full p-1 border rounded" value={config.mode ?? 'frequency'}
          onChange={(e) => onChange({ ...config, mode: e.target.value as EncodingConfig['mode'] })}>
          <option value="frequency">Frequency (share of all training rows)</option>
          <option value="count">Count</option>
        </select>
      </label>
      <p className="text-[10px] text-muted-foreground">Missing values learn their own statistic. Unseen categories become zero.</p>
    </div>
  );
}

function RareOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border">
      <ValidationField field="min_count">
      <label className="block text-xs">Minimum count
        <input aria-label="Minimum count" type="number" min="1" step="1" className="w-full p-1 border rounded"
          value={config.min_count ?? 2} onChange={(e) => onChange({ ...config, min_count: Number(e.target.value) })} />
      </label>
      </ValidationField>
      <ValidationField field="min_share">
      <label className="block text-xs">Minimum share
        <input aria-label="Minimum share" type="number" min="0" max="1" step="0.01" className="w-full p-1 border rounded"
          value={config.min_share ?? 0} onChange={(e) => onChange({ ...config, min_share: Number(e.target.value) })} />
      </label>
      </ValidationField>
      <ValidationField field="replacement">
      <label className="block text-xs">Replacement label
        <input aria-label="Replacement label" type="text" className="w-full p-1 border rounded"
          value={config.replacement ?? 'Other'} onChange={(e) => onChange({ ...config, replacement: e.target.value })} />
      </label>
      </ValidationField>
      <p className="text-[10px] text-muted-foreground">Text categories must meet both thresholds. Frequent missing values stay null; rare and unseen values use the label. Cast numeric categories to text first.</p>
    </div>
  );
}

function WoeOptions({ config, onChange, schema }: EncodingMethodOptionsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <span className="block text-sm font-medium">Target Column</span>
      <ValidationField field="target_column">
        <select
          aria-label="Target Column"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.target_column || ''}
          onChange={(e) => onChange({ ...config, target_column: e.target.value })}
        >
          <option value="">Select a binary target column...</option>
          {Object.keys(schema?.columns ?? {}).map(name => {
            const col = schema?.columns[name];
            return (
              <option key={name} value={name}>{name}{col ? ` (${col.dtype})` : ''}</option>
            );
          })}
        </select>
      </ValidationField>
      <p className="text-[10px] text-muted-foreground">Target must have exactly 2 classes.</p>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Regularization</span>
        <input
          aria-label="Regularization"
          type="number"
          step="0.1"
          min="0"
          className="w-full p-2 border rounded"
          value={config.regularization ?? 0.5}
          onChange={(e) => onChange({ ...config, regularization: Number.parseFloat(e.target.value) })}
          title="Laplace smoothing added to event/non-event counts to avoid division by zero."
        />
        <p className="text-[10px] text-muted-foreground">Smoothing for rare categories (default 0.5).</p>
      </div>
    </div>
  );
}

function HashOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <span className="block text-xs font-medium">Number of Buckets</span>
      <input
        aria-label="Number of Buckets"
        type="number"
        min="1"
        className="w-full p-2 border rounded"
        value={config.n_features ?? 8}
        onChange={(e) => onChange({ ...config, n_features: parseIntSafe(e.target.value, config.n_features) })}
        title="Number of hash buckets."
      />
      <p className="text-[10px] text-muted-foreground">One integer bucket index per source column, from 0 to buckets − 1 (default 8). Categories can collide.</p>
    </div>
  );
}

function OrdinalOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <div className="space-y-1">
        <span className="block text-xs font-medium">Handle Unknown</span>
        <select
          aria-label="Handle Unknown"
          className="w-full p-1 text-sm border rounded"
          value={config.handle_unknown === 'ignore' || !config.handle_unknown ? 'use_encoded_value' : config.handle_unknown}
          onChange={(e) => onChange({ ...config, handle_unknown: e.target.value as EncodingConfig['handle_unknown'] })}
          title="How to handle categories not seen during training."
        >
          <option value="use_encoded_value">Use encoded value (–1)</option>
          <option value="error">Raise error</option>
        </select>
        <p className="text-[10px] text-muted-foreground">
          &quot;Use encoded value&quot; assigns the integer below to unknown categories.
        </p>
      </div>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Unknown Value</span>
        <input
          aria-label="Unknown Value"
          type="number"
          className="w-full p-2 border rounded"
          value={config.unknown_value ?? -1}
          disabled={config.handle_unknown === 'error'}
          onChange={(e) => onChange({ ...config, unknown_value: parseIntSafe(e.target.value, config.unknown_value) })}
          title="Integer to assign for unknown categories."
        />
      </div>
      <div className="space-y-1">
        <span className="block text-xs font-medium">
          Category Order <span className="font-normal text-muted-foreground">(optional)</span>
        </span>
        <textarea
          aria-label="Category Order (optional)"
          className="w-full p-1.5 text-xs border rounded font-mono resize-y min-h-[56px]"
          placeholder={"One line per column:\nlow, medium, high\ncat1, cat2, cat3"}
          value={config.categories_order || ''}
          rows={3}
          onChange={(e) => onChange({ ...config, categories_order: e.target.value })}
          title="Ordered categories per selected column. One line per column (in selection order). Leave empty for auto-detection."
        />
        <p className="text-[10px] text-muted-foreground">
          One line per selected column, comma-separated in desired order. Leave blank for auto-detection.
        </p>
      </div>
    </div>
  );
}

function LabelOptions({ config, onChange }: EncodingSettingsProps) {
  return (
    <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
      <span className="block text-sm font-medium">Missing/Unknown Code</span>
      <input
        aria-label="Missing/Unknown Code"
        type="number"
        className="w-full p-2 border rounded"
        value={config.missing_code ?? -1}
        onChange={(e) => onChange({ ...config, missing_code: parseIntSafe(e.target.value, config.missing_code) })}
        title="Integer to assign for missing or unknown categories."
      />
      <p className="text-xs text-muted-foreground">
        Value assigned to missing/unknown categories (default: -1).
      </p>
    </div>
  );
}

const methodOptions = {
  onehot: OnehotOptions,
  dummy: DummyOptions,
  target: TargetOptions,
  woe: WoeOptions,
  hash: HashOptions,
  ordinal: OrdinalOptions,
  label: LabelOptions,
  frequency: FrequencyOptions,
  rare: RareOptions,
};

export function EncodingMethodOptions(props: EncodingMethodOptionsProps) {
  const Options = Object.prototype.hasOwnProperty.call(methodOptions, props.config.method)
    ? methodOptions[props.config.method] : null;
  return <div className="space-y-2">{Options ? <Options {...props} /> : null}</div>;
}
