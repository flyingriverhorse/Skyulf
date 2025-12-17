import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { AlertTriangle, Search, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

// --- Types ---

interface InvalidValueReplacementConfig {
  columns: string[];
  mode: 'negative_to_nan' | 'zero_to_nan' | 'percentage_bounds' | 'age_bounds' | 'custom_range';
  min_value?: number;
  max_value?: number;
}

// --- Components ---

const ColumnSelector: React.FC<{
  columns: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
}> = ({ columns, selected, onChange }) => {
  const [search, setSearch] = useState('');
  
  const filtered = columns.filter(c => c.toLowerCase().includes(search.toLowerCase()));

  const toggle = (col: string) => {
    if (selected.includes(col)) {
      onChange(selected.filter(c => c !== col));
    } else {
      onChange([...selected, col]);
    }
  };

  return (
    <div className="border rounded bg-background overflow-hidden flex flex-col max-h-40">
      <div className="flex items-center px-2 py-1.5 border-b bg-muted/20">
        <Search size={12} className="text-muted-foreground mr-1.5" />
        <input
          className="flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground/70"
          placeholder="Search columns..."
          value={search}
          onChange={e => { setSearch(e.target.value); }}
        />
      </div>
      <div className="overflow-y-auto p-1 space-y-0.5">
        {filtered.length > 0 ? (
          filtered.map(col => {
            const isSelected = selected.includes(col);
            return (
              <div
                key={col}
                onClick={() => { toggle(col); }}
                className={`
                  flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors
                  ${isSelected ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 font-medium' : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'}
                `}
              >
                <div className={`
                  w-3 h-3 rounded border flex items-center justify-center
                  ${isSelected ? 'border-blue-500 bg-blue-500 text-white' : 'border-gray-400 dark:border-gray-600'}
                `}>
                  {isSelected && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                </div>
                <span className="truncate">{col}</span>
              </div>
            );
          })
        ) : (
          <div className="p-2 text-xs text-gray-500 text-center italic">No columns found</div>
        )}
      </div>
    </div>
  );
};

const InvalidValueSettings: React.FC<{ config: InvalidValueReplacementConfig; onChange: (c: InvalidValueReplacementConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  const [showInfo, setShowInfo] = useState(true);

  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find((d: unknown) => (d as Record<string, any>).datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  
  // Filter for numeric columns only
  const numericColumns = schema 
    ? Object.values(schema.columns)
        .filter((c: unknown) => ['int', 'float', 'number'].some(t => (c as Record<string, any>).dtype?.toLowerCase().includes(t)))
        .map((c: unknown) => (c as Record<string, any>).name)
    : [];

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > 400);
      }
    });
    observer.observe(containerRef.current);
    return () => { observer.disconnect(); };
  }, []);

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-white dark:bg-gray-900 ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column: Column Selection */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
              Target Columns ({config.columns.length})
            </label>
            <ColumnSelector 
              columns={numericColumns} 
              selected={config.columns} 
              onChange={(cols) => onChange({ ...config, columns: cols })} 
            />
            <p className="text-xs text-gray-500 mt-1">Only numeric columns are shown.</p>
          </div>
        </div>

        {/* Right Column: Settings */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
            <button 
              onClick={() => { setShowInfo(!showInfo); }}
              className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
              <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
                About Invalid Value Replacement
              </span>
              {showInfo ? (
                <ChevronUp size={14} className="text-blue-600 dark:text-blue-400" />
              ) : (
                <ChevronDown size={14} className="text-blue-600 dark:text-blue-400" />
              )}
            </button>
            
            {showInfo && (
              <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
                <p>Use this node to handle data quality issues by replacing invalid numeric values (like negative ages or out-of-bounds sensor readings) with NaN (Missing).</p>
              </div>
            )}
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
              Replacement Mode
            </label>
            <select
              className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2"
              value={config.mode}
              onChange={(e) => onChange({ ...config, mode: e.target.value as InvalidValueReplacementConfig['mode'] })}
            >
              <option value="negative_to_nan">Negative to NaN</option>
              <option value="zero_to_nan">Zero to NaN</option>
              <option value="percentage_bounds">Percentage Bounds (0-100)</option>
              <option value="age_bounds">Age Bounds (0-120)</option>
              <option value="custom_range">Custom Range</option>
            </select>
            <p className="text-[10px] text-gray-500 mt-1">
              {config.mode === 'negative_to_nan' && "Replaces all negative values (< 0) with NaN."}
              {config.mode === 'zero_to_nan' && "Replaces all zero values (== 0) with NaN."}
              {config.mode === 'percentage_bounds' && "Replaces values outside 0-100 range with NaN."}
              {config.mode === 'age_bounds' && "Replaces values outside 0-120 range with NaN."}
              {config.mode === 'custom_range' && "Replaces values outside the specified Min/Max range with NaN."}
            </p>
          </div>

          {(config.mode === 'custom_range' || config.mode === 'percentage_bounds' || config.mode === 'age_bounds') && (
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-gray-500 uppercase font-semibold">Min Value</label>
                <input
                  type="number"
                  className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1.5"
                  placeholder={config.mode === 'percentage_bounds' ? '0' : 'Min'}
                  value={config.min_value ?? ''}
                  onChange={(e) => onChange({ ...config, min_value: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </div>
              <div>
                <label className="text-[10px] text-gray-500 uppercase font-semibold">Max Value</label>
                <input
                  type="number"
                  className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1.5"
                  placeholder={config.mode === 'percentage_bounds' ? '100' : 'Max'}
                  value={config.max_value ?? ''}
                  onChange={(e) => onChange({ ...config, max_value: e.target.value ? parseFloat(e.target.value) : undefined })}
                />
              </div>
              <p className="col-span-2 text-[10px] text-gray-500">
                Values outside this range will be replaced with NaN.
              </p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export const InvalidValueReplacementNode: NodeDefinition<InvalidValueReplacementConfig> = {
  type: 'InvalidValueReplacement',
  label: 'Invalid Value Replacement',
  category: 'Preprocessing',
  description: 'Replace invalid or out-of-range values with NaN.',
  icon: AlertTriangle,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: InvalidValueSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column.' };
    if (config.mode === 'custom_range' && config.min_value === undefined && config.max_value === undefined) {
      return { isValid: false, message: 'Specify at least a min or max value for custom range.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    mode: 'negative_to_nan',
  }),
};
