import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Scaling } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface ScalingConfig {
  columns: string[];
  method: 'standard' | 'minmax' | 'maxabs' | 'robust';
  
  // Standard Scaler
  with_mean?: boolean;
  with_std?: boolean;

  // MinMax Scaler
  feature_range_min?: number;
  feature_range_max?: number;

  // Robust Scaler
  quantile_range_min?: number;
  quantile_range_max?: number;
  with_centering?: boolean;
  with_scaling?: boolean;
}

const ScalingSettings: React.FC<{ config: ScalingConfig; onChange: (c: ScalingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  
  // Responsive Layout
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

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

  // Filter for numeric columns only, as scaling only applies to them
  const numericColumns = schema 
    ? Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];

  const filteredColumns = numericColumns.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));

  const handleSelectAll = () => {
    onChange({ ...config, columns: filteredColumns });
  };

  const handleDeselectAll = () => {
    const newCols = config.columns.filter(c => !filteredColumns.includes(c));
    onChange({ ...config, columns: newCols });
  };

  const FeedbackSection = () => {
    if (!nodeResult || !nodeResult.metrics) return null;
    const m = nodeResult.metrics;
    const cols = m.columns as string[] | undefined;
    
    if (!cols) return null;

    const getTargetExplanation = () => {
      switch (config.method) {
        case 'standard': return "Target: μ ≈ 0, σ ≈ 1";
        case 'minmax': return `Target: ${config.feature_range_min ?? 0} to ${config.feature_range_max ?? 1}`;
        case 'robust': return "Target: Median ≈ 0, IQR ≈ 1";
        case 'maxabs': return "Target: MaxAbs = 1";
        default: return "";
      }
    };

    return (
      <div className="mt-4 p-3 bg-muted/50 rounded border text-xs space-y-2">
        <div className="flex justify-between items-center">
          <div className="font-medium text-muted-foreground">Scaling Statistics</div>
          <div className="text-[10px] text-primary/80 bg-primary/5 px-1.5 py-0.5 rounded border border-primary/10" title="Values close to these targets indicate successful scaling.">
             {getTargetExplanation()}
          </div>
        </div>
        <div className="max-h-40 overflow-y-auto bg-background p-2 rounded border space-y-1">
          {cols.map((col, idx) => {
            let details = "";
            if (config.method === 'standard') {
               const mean = (m.mean as number[])[idx];
               const scale = (m.scale as number[])[idx];
               if (typeof mean === 'number') details = `μ=${mean.toFixed(2)}, σ=${typeof scale === 'number' ? scale.toFixed(2) : '-'}`;
            } else if (config.method === 'minmax') {
               const min = (m.data_min as number[])[idx];
               const max = (m.data_max as number[])[idx];
               if (typeof min === 'number') details = `Min=${min.toFixed(2)}, Max=${typeof max === 'number' ? max.toFixed(2) : '-'}`;
            } else if (config.method === 'robust') {
               const center = (m.center as number[])[idx];
               const scale = (m.scale as number[])?.[idx];
               if (typeof center === 'number') details = `Med=${center.toFixed(2)}, IQR=${typeof scale === 'number' ? scale.toFixed(2) : '-'}`;
            } else if (config.method === 'maxabs') {
               const maxAbs = (m.max_abs as number[])[idx];
               if (typeof maxAbs === 'number') details = `MaxAbs=${maxAbs.toFixed(2)}`;
            }

            return (
              <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
                <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
                <span className="font-mono text-[10px] text-muted-foreground">{details}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!datasetId && (
          <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
            Connect a dataset node to see available columns.
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column: Settings */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div>
            <label className="block text-sm font-medium mb-1">Scaling Method</label>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.method}
              onChange={(e) => onChange({ ...config, method: e.target.value as any })}
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

          {/* Standard Scaler Params */}
          {config.method === 'standard' && (
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
          )}

          {/* MinMax Scaler Params */}
          {config.method === 'minmax' && (
            <div className="space-y-2 border-t pt-2">
              <label className="block text-sm font-medium">Feature Range</label>
              <div className="flex gap-2 items-center">
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  placeholder="Min (0)"
                  value={config.feature_range_min ?? 0}
                  onChange={(e) => onChange({ ...config, feature_range_min: parseFloat(e.target.value) })}
                />
                <span className="text-muted-foreground">-</span>
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  placeholder="Max (1)"
                  value={config.feature_range_max ?? 1}
                  onChange={(e) => onChange({ ...config, feature_range_max: parseFloat(e.target.value) })}
                />
              </div>
            </div>
          )}

          {/* Robust Scaler Params */}
          {config.method === 'robust' && (
            <div className="space-y-2 border-t pt-2">
              <label className="block text-sm font-medium">Quantile Range</label>
              <div className="flex gap-2 items-center">
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  placeholder="Min (25.0)"
                  value={config.quantile_range_min ?? 25.0}
                  onChange={(e) => onChange({ ...config, quantile_range_min: parseFloat(e.target.value) })}
                />
                <span className="text-muted-foreground">-</span>
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  placeholder="Max (75.0)"
                  value={config.quantile_range_max ?? 75.0}
                  onChange={(e) => onChange({ ...config, quantile_range_max: parseFloat(e.target.value) })}
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
          )}
          
          {/* Feedback Section (Wide) */}
          {isWide && <FeedbackSection />}
        </div>
        
        {/* Right Column: Columns */}
        <div className={`flex flex-col h-full min-h-[200px] border rounded-md overflow-hidden ${isWide ? '' : 'shrink-0'}`}>
           <div className="p-2 border-b bg-muted/30 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Numeric Columns ({config.columns.length})</span>
                <div className="flex gap-1">
                  <button onClick={handleSelectAll} className="text-[10px] px-2 py-1 hover:bg-accent rounded">All</button>
                  <button onClick={handleDeselectAll} className="text-[10px] px-2 py-1 hover:bg-accent rounded">None</button>
                </div>
              </div>
              <input
                type="text"
                placeholder="Search columns..."
                className="w-full text-xs p-1.5 border rounded bg-background"
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); }}
              />
           </div>
           <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredColumns.length > 0 ? (
                filteredColumns.map(col => (
                  <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-1.5 rounded cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={config.columns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...config.columns, col]
                          : config.columns.filter(c => c !== col);
                        onChange({ ...config, columns: newCols });
                      }}
                      className="rounded border-gray-300 text-primary focus:ring-primary"
                    />
                    <span className="truncate" title={col}>{col}</span>
                  </label>
                ))
            ) : (
              <div className="text-xs text-muted-foreground italic p-4 text-center">
                {isLoading ? 'Loading schema...' : (numericColumns.length === 0 ? 'No numeric columns found' : 'No matches found')}
              </div>
            )}
           </div>
        </div>

        {/* Feedback Section (Narrow) */}
        {!isWide && <FeedbackSection />}

      </div>
    </div>
  );
};

export const ScalingNode: NodeDefinition<ScalingConfig> = {
  type: 'scale_numeric_features',
  label: 'Scaling',
  category: 'Preprocessing',
  description: 'Scale numeric features to a standard range.',
  icon: Scaling, // Note: You might need to import a real icon or use a placeholder if 'Scaling' doesn't exist in lucide-react
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Scaled Data', type: 'dataset' }],
  settings: ScalingSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0,
    message: config.columns.length === 0 ? 'Select at least one column' : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
    method: 'standard',
  }),
};
