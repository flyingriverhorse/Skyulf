import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { BarChart3 } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';

interface BinningConfig {
  columns: string[];
  strategy: 'equal_width' | 'equal_frequency' | 'kmeans' | 'custom';
  n_bins: number;
  label_format: 'ordinal' | 'range' | 'bin_index';
  output_suffix?: string;
  drop_original?: boolean;
  custom_bins?: Record<string, number[]>; // Map column -> edges
  custom_labels?: Record<string, string[]>; // Map column -> labels
  precision?: number;
}

const CustomBinInput: React.FC<{
  value: number[] | undefined;
  onChange: (val: number[]) => void;
}> = ({ value, onChange }) => {
  const [text, setText] = useState(value?.join(', ') || '');
  const lastValueRef = useRef(value);

  useEffect(() => {
    if (JSON.stringify(value) !== JSON.stringify(lastValueRef.current)) {
      setText(value?.join(', ') || '');
      lastValueRef.current = value;
    }
  }, [value]);

  const handleBlur = () => {
    const edges = text.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n)).sort((a, b) => a - b);
    onChange(edges);
  };

  return (
    <input
      type="text"
      className="w-full p-1 text-xs border rounded bg-background text-foreground"
      placeholder="0, 10, 20, 100..."
      value={text}
      onChange={(e) => { setText(e.target.value); }}
      onBlur={handleBlur}
    />
  );
};

const BinningSettings: React.FC<{ config: BinningConfig; onChange: (c: BinningConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);

  // Responsive Layout
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > 450);
      }
    });
    observer.observe(containerRef.current);
    return () => { observer.disconnect(); };
  }, []);

  // Filter for numeric columns only
  const numericColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
        .filter(c => !droppedUpstream.has(c.name))
        .map(c => c.name)
    : [];

  return (
    <div ref={containerRef} className="p-4 space-y-4 h-full overflow-y-auto">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
          Connect a dataset node to see available columns.
        </div>
      )}

      <div className={`grid gap-4 ${isWide ? 'grid-cols-2' : 'grid-cols-1'}`}>
        {/* Left Column: Settings */}
        <div className="space-y-4">
          <div className="space-y-2">
            <span className="block text-sm font-medium">Binning Strategy</span>
            <select
              className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
              value={config.strategy}
              onChange={(e) => onChange({ ...config, strategy: e.target.value as BinningConfig['strategy'] })}
            >
              <option value="equal_width">Equal Width (Uniform)</option>
              <option value="equal_frequency">Equal Frequency (Quantile)</option>
              <option value="kmeans">K-Means Clustering</option>
              <option value="custom">Custom Edges</option>
            </select>
            <p className="text-[10px] text-muted-foreground">
              {config.strategy === 'equal_width' && "Bins have equal width ranges."}
              {config.strategy === 'equal_frequency' && "Bins have equal number of records."}
              {config.strategy === 'kmeans' && "Bins based on K-Means clustering centers."}
              {config.strategy === 'custom' && "Define specific bin edges for each column."}
            </p>
          </div>

          {config.strategy !== 'custom' && (
            <div className="space-y-2">
              <span className="block text-sm font-medium">Number of Bins</span>
              <input
                type="number"
                min={2}
                max={100}
                className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
                value={config.n_bins}
                onChange={(e) => onChange({ ...config, n_bins: parseInt(e.target.value) || 5 })}
              />
            </div>
          )}

          {config.strategy === 'custom' && (
             <div className="space-y-2 border rounded p-2 bg-muted/10">
               <span className="block text-xs font-medium">Custom Edges</span>
               {config.columns.length === 0 ? (
                 <div className="text-xs text-muted-foreground">Select columns first.</div>
               ) : (
                 <div className="space-y-2 max-h-40 overflow-y-auto">
                   {config.columns.map(col => (
                     <div key={col} className="space-y-1">
                       <label className="text-[10px] font-medium">{col}</label>
                       <CustomBinInput
                         value={config.custom_bins?.[col]}
                         onChange={(edges) => onChange({
                           ...config,
                           custom_bins: { ...config.custom_bins, [col]: edges }
                         })}
                       />
                     </div>
                   ))}
                 </div>
               )}
             </div>
          )}

          <div className="space-y-2">
            <span className="block text-sm font-medium">Label Format</span>
            <select
              className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
              value={config.label_format}
              onChange={(e) => onChange({ ...config, label_format: e.target.value as BinningConfig['label_format'] })}
            >
              <option value="ordinal">Ordinal (0, 1, 2...)</option>
              <option value="range">Range Intervals ([0-10], [10-20]...)</option>
              <option value="bin_index">Bin Index</option>
            </select>
          </div>

          {config.label_format === 'range' && (
            <div className="space-y-2">
              <span className="block text-sm font-medium">Precision (Decimals)</span>
              <input
                type="number"
                min={0}
                max={10}
                className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
                value={config.precision ?? 3}
                onChange={(e) => onChange({ ...config, precision: parseInt(e.target.value) || 0 })}
              />
              <p className="text-[10px] text-muted-foreground">
                Number of decimal places to show in interval labels (e.g. 2 gives [0.00, 10.00]).
              </p>
            </div>
          )}

          <div className="space-y-2 pt-2 border-t">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="drop_original"
                checked={config.drop_original}
                onChange={(e) => onChange({ ...config, drop_original: e.target.checked })}
                className="rounded border-gray-300 text-primary focus:ring-primary"
              />
              <label htmlFor="drop_original" className="text-sm">Drop Original Columns</label>
            </div>

            {!config.drop_original && (
              <div className="space-y-1">
                <span className="text-xs font-medium text-muted-foreground">Output Suffix</span>
                <input
                  type="text"
                  className="w-full p-1.5 border rounded text-xs"
                  placeholder="_binned"
                  value={config.output_suffix || ''}
                  onChange={(e) => onChange({ ...config, output_suffix: e.target.value })}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Column Selection */}
        <ColumnMultiSelect
          columns={numericColumns}
          selected={config.columns}
          onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
          label="Target Columns"
          variant="panel"
          isLoading={isLoading}
          emptyMessage="No numeric columns found."
          fillHeight={false}
        />
      </div>
    </div>
  );
};

export const BinningNode: NodeDefinition = {
  type: 'BinningNode',
  label: 'Binning / Discretization',
  description: 'Convert continuous variables into discrete bins.',
  icon: BarChart3,
  category: 'Preprocessing',
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Binned' }],
  validate: (config: BinningConfig) => {
    if (config.columns.length === 0) {
      return { isValid: false, message: 'Select at least one column to bin.' };
    }
    if (config.n_bins < 2) {
      return { isValid: false, message: 'Number of bins must be at least 2.' };
    }
    return { isValid: true };
  },
  settings: BinningSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const strat = config.strategy ?? 'equal_width';
    const n = config.n_bins ?? 5;
    if (cols === 0) return `${strat} · q=${n}`;
    return `${strat} · q=${n} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  getDefaultConfig: () => ({
    columns: [],
    strategy: 'equal_width',
    n_bins: 5,
    label_format: 'ordinal',
    output_suffix: '_binned',
    drop_original: false
  })
};
