import { ValidationField } from '../../../../components/shared/ValidationField';
import type { ManualColumnBounds, OutlierSettingsProps } from './types';

/** Keep an omitted endpoint empty, and preserve malformed numeric input for validation. */
function readBound(input: HTMLInputElement): number | null {
  if (input.validity.badInput) return NaN;
  return input.value === '' ? null : Number(input.value);
}

function BoundInput({ config, onChange, column, endpoint }: OutlierSettingsProps & {
  column: string; endpoint: keyof ManualColumnBounds;
}) {
  const bound = config.bounds?.[column] ?? {};
  const value = bound[endpoint];
  const label = endpoint === 'lower' ? 'Lower' : 'Upper';
  return <ValidationField field={`bounds.${column}.${endpoint}`}>
    <label className="block space-y-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <input
        aria-label={`${label} bound for ${column}`}
        type="number"
        step="any"
        placeholder="No limit"
        className="w-full min-w-0 p-2 border rounded bg-background text-sm"
        value={Number.isFinite(value) ? value as number : ''}
        onChange={(event) => onChange({
          ...config,
          bounds: { ...config.bounds, [column]: { ...bound, [endpoint]: readBound(event.currentTarget) } },
        })}
      />
    </label>
  </ValidationField>;
}

/** Configure only selected columns while retaining inactive bounds in graph settings. */
export function ManualBoundsOptions({ config, onChange }: OutlierSettingsProps) {
  return <ValidationField field="bounds" className="space-y-3">
    <p className="text-xs text-muted-foreground">
      Set at least one bound per selected column. Leave the other blank for no limit. Endpoints are inclusive.
    </p>
    {config.columns.map((column) => <fieldset key={column} className="min-w-0 border rounded p-3">
      <legend className="px-1 text-xs font-medium break-all">{column}</legend>
      <div className="grid grid-cols-2 gap-2">
        <BoundInput config={config} onChange={onChange} column={column} endpoint="lower" />
        <BoundInput config={config} onChange={onChange} column={column} endpoint="upper" />
      </div>
      <button
        type="button"
        aria-label={`Remove bounds for ${column}`}
        className="mt-2 text-xs text-muted-foreground hover:text-foreground underline underline-offset-2"
        onClick={() => onChange({ ...config, columns: config.columns.filter(selected => selected !== column) })}
      >Remove</button>
    </fieldset>)}
  </ValidationField>;
}
