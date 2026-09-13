import { useId } from 'react';
import { ValidationField } from '../../../../components/shared/ValidationField';
import { useIsWideContainer } from '../../../../core/hooks/useIsWideContainer';
import type { NodeSettingsProps } from '../../../../core/types/nodes';
import { COORDINATES, type GeoDistanceConfig } from './types';
import { useGeoDistanceColumns } from './useGeoDistanceColumns';

const controlClass = 'w-full min-w-0 px-2 py-1.5 text-sm border rounded bg-background';

/** Keep unavailable saved references visible until the user chooses a replacement. */
function CoordinateSelect({ label, field, value, columns, isLoading, onChange }: {
  label: string;
  field: string;
  value: string;
  columns: string[];
  isLoading: boolean;
  onChange: (column: string) => void;
}) {
  const id = useId();
  return <ValidationField field={field} className="space-y-1">
    <label htmlFor={id} className="block text-xs font-medium">{label}</label>
    <select id={id} className={controlClass} value={value} disabled={isLoading}
      onChange={event => onChange(event.target.value)}>
      <option value="">Select a numeric column</option>
      {value && !columns.includes(value) && <option value={value} disabled>{value} (unavailable)</option>}
      {columns.map(column => <option key={column} value={column}>{column}</option>)}
    </select>
  </ValidationField>;
}

/** Explain the geographic input contract alongside each distance method. */
function DistanceControls({ config, onChange }: NodeSettingsProps<GeoDistanceConfig>) {
  const id = useId();
  return <div className="min-w-0 space-y-4">
    <ValidationField field="method" className="space-y-1">
      <label htmlFor={`${id}-method`} className="block text-xs font-medium">Distance method</label>
      <select id={`${id}-method`} aria-describedby={`${id}-method-help`} className={controlClass}
        value={config.method} onChange={event => onChange({ ...config, method: event.target.value as GeoDistanceConfig['method'] })}>
        <option value="haversine">Haversine (great-circle)</option>
        <option value="euclidean">Euclidean (local approximation)</option>
      </select>
      <p id={`${id}-method-help`} className="text-xs text-muted-foreground">
        {config.method === 'euclidean'
          ? 'Flat-earth approximation for nearby points. Use Haversine for longer distances or points across the date line.'
          : 'Great-circle distance along the Earth\'s surface. Recommended for geographic coordinates.'}
      </p>
    </ValidationField>
    <ValidationField field="unit" className="space-y-1">
      <label htmlFor={`${id}-unit`} className="block text-xs font-medium">Distance unit</label>
      <select id={`${id}-unit`} className={controlClass} value={config.unit}
        onChange={event => onChange({ ...config, unit: event.target.value as GeoDistanceConfig['unit'] })}>
        <option value="km">Kilometres (km)</option>
        <option value="mi">Miles (mi)</option>
      </select>
    </ValidationField>
    <ValidationField field="output_column" className="space-y-1">
      <label htmlFor={`${id}-output`} className="block text-xs font-medium">Output column (optional)</label>
      <input id={`${id}-output`} type="text" className={controlClass} value={config.output_column ?? ''}
        placeholder={`geo_distance_${config.unit}`} aria-describedby={`${id}-output-help`}
        onChange={event => onChange({ ...config, output_column: event.target.value })} />
      <p id={`${id}-output-help`} className="text-xs text-muted-foreground">
        Leave blank to use <span className="break-all">geo_distance_{config.unit}</span>.
        {' '}An existing column with this name is replaced; other columns are kept.
      </p>
    </ValidationField>
  </div>;
}

/** Render four numeric coordinate inputs in the compact or expanded Properties panel. */
export default function GeoDistanceSettings({ config, onChange, nodeId }: NodeSettingsProps<GeoDistanceConfig>) {
  const { datasetId, numericColumns, isLoading } = useGeoDistanceColumns(nodeId);
  const [containerRef, isWide] = useIsWideContainer();
  return <div ref={containerRef} className="w-full min-w-0 p-4 space-y-4">
    <p className="text-xs text-muted-foreground">
      Choose latitude and longitude in degrees for each point. Each row produces one distance.
    </p>
    {!datasetId && <p className="text-xs text-muted-foreground">Connect a dataset node to see available columns.</p>}
    {isLoading && <p role="status" className="text-xs text-muted-foreground">Loading columns...</p>}
    {!isLoading && numericColumns.length === 0 && <p className="text-xs text-muted-foreground">No numeric columns available.</p>}
    <div className={`grid gap-4 ${isWide ? 'grid-cols-2' : 'grid-cols-1'}`}>
      <div className="min-w-0 space-y-4">
        {COORDINATES.map(({ field, label }) => <CoordinateSelect key={field} field={field} label={label}
          columns={numericColumns} value={config[field] ?? ''} isLoading={isLoading}
          onChange={column => onChange({ ...config, [field]: column })} />)}
      </div>
      <DistanceControls config={config} onChange={onChange} />
    </div>
  </div>;
}
