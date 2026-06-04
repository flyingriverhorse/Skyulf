import { useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Clock, Search, Info } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { clickableProps } from '../../../core/utils/a11y';

type TimeSeriesMethod = 'lag' | 'rolling' | 'date';

interface TimeSeriesConfig {
  method: TimeSeriesMethod;
  columns: string[];
  // Ordering / grouping (shared by lag + rolling)
  sort_by?: string | undefined;
  group_by?: string[] | undefined;
  // Lag
  lags?: number[] | undefined;
  drop_na?: boolean | undefined;
  // Rolling
  window?: number | undefined;
  aggregations?: string[] | undefined;
  min_periods?: number | undefined;
  // Date
  features?: string[] | undefined;
  drop_original?: boolean | undefined;
}

const ROLLING_AGGS = ['mean', 'sum', 'min', 'max', 'std', 'median'];
const DATE_FEATURES = [
  'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
  'weekofyear', 'hour', 'minute', 'is_weekend', 'is_month_start', 'is_month_end',
];

function ColumnSelector({ columns, selected, onChange, label }: {
  columns: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  label?: string;
}) {
  const [search, setSearch] = useState('');
  const filtered = columns.filter(c => c.toLowerCase().includes(search.toLowerCase()));

  const toggle = (col: string) => {
    if (selected.includes(col)) onChange(selected.filter(c => c !== col));
    else onChange([...selected, col]);
  };

  return (
    <div className="space-y-1.5">
      {label && <span className="block text-xs font-medium text-muted-foreground">{label}</span>}
      <div className="border rounded bg-background overflow-hidden flex flex-col">
        <div className="flex items-center px-2 py-1.5 border-b bg-muted/20">
          <Search size={12} className="text-muted-foreground mr-1.5" />
          <input
            className="flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground/70"
            placeholder="Search columns..."
            value={search}
            onChange={e => { setSearch(e.target.value); }}
          />
        </div>
        <div className="max-h-32 overflow-y-auto p-1 space-y-0.5">
          {filtered.length > 0 ? (
            filtered.map(col => {
              const isSelected = selected.includes(col);
              return (
                <div
                  key={col}
                  {...clickableProps(() => { toggle(col); })}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors ${isSelected ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-accent text-foreground'}`}
                >
                  <div className={`w-3 h-3 rounded border flex items-center justify-center ${isSelected ? 'border-primary bg-primary text-primary-foreground' : 'border-muted-foreground/40'}`}>
                    {isSelected && <div className="w-1.5 h-1.5 bg-current rounded-sm" />}
                  </div>
                  <span className="truncate">{col}</span>
                </div>
              );
            })
          ) : (
            <div className="p-2 text-xs text-muted-foreground text-center">No columns found</div>
          )}
        </div>
      </div>
      <div className="text-[10px] text-muted-foreground text-right">{selected.length} selected</div>
    </div>
  );
}

function ChipMultiSelect({ options, selected, onChange }: {
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}) {
  const toggle = (opt: string) => {
    if (selected.includes(opt)) onChange(selected.filter(o => o !== opt));
    else onChange([...selected, opt]);
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map(opt => {
        const active = selected.includes(opt);
        return (
          <button
            key={opt}
            type="button"
            onClick={() => { toggle(opt); }}
            className={`px-2 py-1 rounded text-[11px] border transition-colors ${active ? 'bg-primary text-primary-foreground border-primary' : 'bg-background hover:bg-accent border-muted-foreground/30'}`}
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

const METHOD_DESCRIPTIONS: Record<TimeSeriesMethod, string> = {
  lag: 'Shift columns by N rows to expose past values to the model.',
  rolling: 'Rolling-window aggregates (mean/sum/min/max/std/median).',
  date: 'Extract calendar parts (year, month, day-of-week, ...) from datetime columns.',
};

type UpdateFn = (updates: Partial<TimeSeriesConfig>) => void;

function OrderingSelectors({ config, allColumns, update }: {
  config: TimeSeriesConfig;
  allColumns: string[];
  update: UpdateFn;
}) {
  return (
  <div className="grid grid-cols-2 gap-2">
    <div className="space-y-1">
      <span className="block text-xs font-medium text-muted-foreground">Sort By (optional)</span>
      <select
        className="w-full p-1.5 border rounded bg-background text-xs"
        value={config.sort_by ?? ''}
        onChange={e => { update({ sort_by: e.target.value || undefined }); }}
      >
        <option value="">— none —</option>
        {allColumns.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
    <div className="space-y-1">
      <span className="block text-xs font-medium text-muted-foreground">Group By (optional)</span>
      <select
        className="w-full p-1.5 border rounded bg-background text-xs"
        value={config.group_by?.[0] ?? ''}
        onChange={e => { update({ group_by: e.target.value ? [e.target.value] : undefined }); }}
      >
        <option value="">— none —</option>
        {allColumns.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
  </div>
  );
}

function LagSettings({ config, update }: { config: TimeSeriesConfig; update: UpdateFn }) {
  return (
  <div className="space-y-2 p-3 bg-muted/20 rounded border">
    <div className="space-y-1">
      <span className="block text-xs font-medium">Lags (comma-separated)</span>
      <input
        type="text"
        className="w-full p-1.5 border rounded text-sm"
        value={(config.lags ?? [1]).join(', ')}
        onChange={e => {
          const parsed = e.target.value
            .split(',')
            .map(s => parseInt(s.trim(), 10))
            .filter(n => Number.isFinite(n));
          update({ lags: parsed });
        }}
        placeholder="1, 2, 3"
      />
      <p className="text-[10px] text-muted-foreground">Each value creates a <code>col_lag_N</code> column.</p>
    </div>
    <label className="flex items-center gap-2 text-xs">
      <input
        type="checkbox"
        checked={config.drop_na ?? false}
        onChange={e => { update({ drop_na: e.target.checked }); }}
      />
      Drop rows with NaN introduced by shifting
    </label>
  </div>
  );
}

function RollingSettings({ config, update }: { config: TimeSeriesConfig; update: UpdateFn }) {
  return (
  <div className="space-y-2 p-3 bg-muted/20 rounded border">
    <div className="grid grid-cols-2 gap-2">
      <div className="space-y-1">
        <span className="block text-xs font-medium">Window</span>
        <input
          type="number"
          min="1"
          className="w-full p-1.5 border rounded text-sm"
          value={config.window ?? 3}
          onChange={e => { update({ window: parseInt(e.target.value, 10) }); }}
        />
      </div>
      <div className="space-y-1">
        <span className="block text-xs font-medium">Min Periods</span>
        <input
          type="number"
          min="1"
          className="w-full p-1.5 border rounded text-sm"
          value={config.min_periods ?? 1}
          onChange={e => { update({ min_periods: parseInt(e.target.value, 10) }); }}
        />
      </div>
    </div>
    <div className="space-y-1">
      <span className="block text-xs font-medium">Aggregations</span>
      <ChipMultiSelect
        options={ROLLING_AGGS}
        selected={config.aggregations ?? ['mean']}
        onChange={aggs => { update({ aggregations: aggs }); }}
      />
    </div>
  </div>
  );
}

function DateSettings({ config, update }: { config: TimeSeriesConfig; update: UpdateFn }) {
  return (
  <div className="space-y-2 p-3 bg-muted/20 rounded border">
    <div className="space-y-1">
      <span className="block text-xs font-medium">Calendar Features</span>
      <ChipMultiSelect
        options={DATE_FEATURES}
        selected={config.features ?? ['year', 'month', 'day', 'dayofweek']}
        onChange={feats => { update({ features: feats }); }}
      />
    </div>
    <label className="flex items-center gap-2 text-xs">
      <input
        type="checkbox"
        checked={config.drop_original ?? false}
        onChange={e => { update({ drop_original: e.target.checked }); }}
      />
      Drop the original datetime column
    </label>
  </div>
  );
}

function TimeSeriesSettings({ config, onChange, nodeId }: {
  config: TimeSeriesConfig;
  onChange: (c: TimeSeriesConfig) => void;
  nodeId?: string;
}) {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  const allColumns = schema
    ? Object.values(schema.columns)
        .filter(col => !droppedUpstream.has(col.name))
        .map(col => col.name)
    : [];

  const update: UpdateFn = (updates) => { onChange({ ...config, ...updates }); };

  return (
    <div className="space-y-3 text-sm">
      <div className="space-y-1.5">
        <span className="block text-xs font-medium text-muted-foreground">Method</span>
        <select
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.method}
          onChange={e => { update({ method: e.target.value as TimeSeriesMethod }); }}
        >
          <option value="lag">Lag Features</option>
          <option value="rolling">Rolling Aggregate</option>
          <option value="date">Date Features</option>
        </select>
        <p className="text-xs text-muted-foreground">{METHOD_DESCRIPTIONS[config.method]}</p>
      </div>

      <ColumnSelector
        columns={allColumns}
        selected={config.columns ?? []}
        onChange={cols => { update({ columns: cols }); }}
        label={config.method === 'date' ? 'Datetime Columns' : 'Numeric Columns'}
      />

      {config.method !== 'date' && (
        <OrderingSelectors config={config} allColumns={allColumns} update={update} />
      )}
      {config.method === 'lag' && <LagSettings config={config} update={update} />}
      {config.method === 'rolling' && <RollingSettings config={config} update={update} />}
      {config.method === 'date' && <DateSettings config={config} update={update} />}

      <div className="flex items-start gap-1.5 text-[10px] text-muted-foreground bg-muted/20 border rounded px-2 py-1.5">
        <Info size={11} className="shrink-0 mt-0.5" />
        <span>Time-series transforms assume rows are ordered. Use <strong>Sort By</strong> to enforce temporal order, and <strong>Group By</strong> to keep entities (e.g. per-store) independent.</span>
      </div>
    </div>
  );
}

function timeSeriesPreview(config: TimeSeriesConfig): string {
  const cols = config.columns?.length ?? 0;
  const method = config.method ?? 'lag';
  if (cols === 0) return method;
  return `${method} \u00b7 ${cols} ${cols === 1 ? 'col' : 'cols'}`;
}

function validateTimeSeries(config: TimeSeriesConfig): { isValid: boolean; error?: string } {
  if ((config.columns?.length ?? 0) === 0)
    return { isValid: false, error: 'Select at least one column' };
  if (config.method === 'lag' && (config.lags?.length ?? 0) === 0)
    return { isValid: false, error: 'Provide at least one lag value' };
  if (config.method === 'rolling' && (config.aggregations?.length ?? 0) === 0)
    return { isValid: false, error: 'Select at least one aggregation' };
  if (config.method === 'date' && (config.features?.length ?? 0) === 0)
    return { isValid: false, error: 'Select at least one calendar feature' };
  return { isValid: true };
}

export const TimeSeriesNode: NodeDefinition<TimeSeriesConfig> = {
  type: 'TimeSeriesNode',
  label: 'Time Series',
  category: 'Preprocessing',
  description: 'Lag, rolling-window, and calendar features for time-series data.',
  icon: Clock,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Featured Data', type: 'dataset' }],
  settings: TimeSeriesSettings,
  bodyPreview: timeSeriesPreview,
  validate: validateTimeSeries,
  getDefaultConfig: () => ({
    method: 'lag',
    columns: [],
    lags: [1],
    drop_na: false,
    window: 3,
    aggregations: ['mean'],
    min_periods: 1,
    features: ['year', 'month', 'day', 'dayofweek'],
    drop_original: false,
  }),
};
