import { Info } from 'lucide-react';
import type { EncodingConfig, EncodingSettingsProps } from './types';

const methodDescriptions: Record<EncodingConfig['method'], string> = {
  onehot: 'Creates binary columns for each category.',
  dummy: 'Like One-Hot but drops first category to avoid collinearity.',
  label: 'Assigns a unique integer to each category.',
  ordinal: 'Encodes categories as ordered integers.',
  target: 'Encodes categories based on target mean.',
  woe: 'Replaces each category with its Weight of Evidence (log-odds of a binary target).',
  hash: 'Maps categories to a fixed number of features.',
};

const featureHint = (
  <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded px-2 py-1">
    <Info size={10} className="shrink-0" />
    <span><strong>Feature columns only</strong> — target is always excluded. Use Label Encoder for the target.</span>
  </div>
);

const methodHints = {
  label: (
    <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded px-2 py-1">
      <Info size={10} className="shrink-0" />
      <span>Encodes <strong>target (y)</strong> by default. Place after Feature/Target Split.</span>
    </div>
  ),
  ordinal: (
    <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded px-2 py-1">
      <Info size={10} className="shrink-0" />
      <span>Target-safe — no columns selected encodes <strong>y</strong>. Use when order matters (low/mid/high).</span>
    </div>
  ),
  target: (
    <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded px-2 py-1">
      <Info size={10} className="shrink-0" />
      <span>Replaces each category with the <strong>mean of the target</strong>. Select the target column on the right.</span>
    </div>
  ),
  woe: (
    <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded px-2 py-1">
      <Info size={10} className="shrink-0" />
      <span>Needs a <strong>binary target</strong> (exactly 2 classes). The target is auto-excluded — use Label Encoder for the target itself. Non-binary targets are <strong>skipped</strong> (columns pass through unchanged), not an error. Records Information Value (IV) per column.</span>
    </div>
  ),
  onehot: featureHint,
  dummy: featureHint,
  hash: featureHint,
};

export function EncodingMethodSelection({ config, onChange, id }: EncodingSettingsProps & { id: string }) {
  return (
    <div className="space-y-2">
      <label htmlFor={`${id}-encoding-method`} className="block text-sm font-medium">Encoding Method</label>
      <select
        id={`${id}-encoding-method`}
        className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
        value={config.method}
        onChange={(e) => onChange({ ...config, method: e.target.value as EncodingConfig['method'] })}
      >
        <option value="onehot">One-Hot Encoding</option>
        <option value="dummy">Dummy Encoding</option>
        <option value="label">Label Encoding</option>
        <option value="ordinal">Ordinal Encoding</option>
        <option value="target">Target Encoding</option>
        <option value="woe">Weight of Evidence (WOE)</option>
        <option value="hash">Hash Encoding</option>
      </select>
      <p className="text-xs text-muted-foreground">
        {Object.prototype.hasOwnProperty.call(methodDescriptions, config.method) && methodDescriptions[config.method]}
      </p>

      {Object.prototype.hasOwnProperty.call(methodHints, config.method) && methodHints[config.method]}

      {/* drop_original only applies to OHE (the only encoder that expands to new columns) */}
      {config.method === 'onehot' && (
      <div className="pt-2">
        <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.drop_original !== false} // Default to true
              onChange={(e) => onChange({ ...config, drop_original: e.target.checked })}
              className="rounded border-gray-300"
            />
            Drop Original Column
        </label>
        <p className="text-[10px] text-muted-foreground ml-5">
            If unchecked, keeps the original column alongside encoded features.
        </p>
      </div>
      )}
    </div>
  );
}
