import React, { useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Zap, Search, ChevronDown, ChevronRight, Info } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

interface PolynomialFeaturesConfig {
  columns: string[];
  degree: number;
  interaction_only: boolean;
  include_bias: boolean;
  output_prefix: string;
  include_input_features: boolean;
  isExpanded?: boolean;
}

const ColumnSelector: React.FC<{
  columns: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  label?: string;
}> = ({ columns, selected, onChange, label }) => {
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
    <div className="space-y-1.5">
      {label && <label className="text-xs font-medium text-muted-foreground">{label}</label>}
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
                  onClick={() => { toggle(col); }}
                  className={`
                    flex items-center gap-2 px-2 py-1.5 rounded text-xs cursor-pointer transition-colors
                    ${isSelected ? 'bg-primary/10 text-primary font-medium' : 'hover:bg-accent text-foreground'}
                  `}
                >
                  <div className={`
                    w-3 h-3 rounded border flex items-center justify-center
                    ${isSelected ? 'border-primary bg-primary text-primary-foreground' : 'border-muted-foreground/40'}
                  `}>
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
      <div className="text-[10px] text-muted-foreground text-right">
        {selected.length} selected
      </div>
    </div>
  );
};

export const PolynomialFeaturesNode: NodeDefinition = {
  type: 'PolynomialFeaturesNode',
  label: 'Polynomial Features',
  description: 'Generate polynomial and interaction features.',
  category: 'Preprocessing',
  icon: Zap,
  inputs: [{ id: 'in', label: 'Input Dataset', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Transformed Data', type: 'dataset' }],
  
  getDefaultConfig: () => ({
    columns: [],
    degree: 2,
    interaction_only: false,
    include_bias: false,
    output_prefix: 'poly',
    include_input_features: false,
    isExpanded: true
  }),

  settings: ({ config, onChange, nodeId }) => {
    const upstreamData = useUpstreamData(nodeId || '');
    const datasetId = upstreamData.find((d: any) => d.datasetId)?.datasetId as string | undefined;
    const { data: schema } = useDatasetSchema(datasetId);
    
    // Filter for numeric columns as polynomial features usually apply to numbers
    const numericColumns = schema ? Object.values(schema.columns)
      .filter((col: any) => ['int', 'float', 'number', 'double', 'long'].some(t => col.dtype.toLowerCase().includes(t)))
      .map((col: any) => col.name) : [];

    const updateConfig = (updates: Partial<PolynomialFeaturesConfig>) => {
      onChange({ ...config, ...updates });
    };

    const toggleExpand = () => {
      updateConfig({ isExpanded: !config.isExpanded });
    };

    return (
      <div className="space-y-2 min-w-[280px]">
        <div className="border rounded-md bg-card">
          <div 
            className="flex items-center justify-between p-2 cursor-pointer hover:bg-accent/50 transition-colors"
            onClick={toggleExpand}
          >
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-primary" />
              <span className="text-sm font-medium">Configuration</span>
            </div>
            {config.isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </div>

          {config.isExpanded && (
            <div className="p-3 space-y-4 border-t">
              
              <ColumnSelector
                label="Input Columns (Numeric)"
                columns={numericColumns}
                selected={config.columns || []}
                onChange={(cols) => updateConfig({ columns: cols })}
              />

              <div className="space-y-3">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                    Degree
                    <div className="group relative">
                      <Info size={10} className="cursor-help" />
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block w-48 p-2 bg-popover text-popover-foreground text-[10px] rounded border shadow-lg z-50">
                        The degree of the polynomial features. Default = 2.
                      </div>
                    </div>
                  </label>
                  <input
                    type="number"
                    min={2}
                    max={5}
                    className="w-full px-2 py-1.5 text-xs border rounded bg-background"
                    value={config.degree || 2}
                    onChange={(e) => updateConfig({ degree: parseInt(e.target.value) || 2 })}
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Output Prefix</label>
                  <input
                    type="text"
                    className="w-full px-2 py-1.5 text-xs border rounded bg-background"
                    value={config.output_prefix || 'poly'}
                    onChange={(e) => updateConfig({ output_prefix: e.target.value })}
                  />
                </div>

                <div className="space-y-2 pt-1">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-muted"
                      checked={config.interaction_only || false}
                      onChange={(e) => updateConfig({ interaction_only: e.target.checked })}
                    />
                    <span className="text-xs">Interaction Only</span>
                  </label>
                  <p className="text-[10px] text-muted-foreground pl-5">
                    If true, only interaction features are produced: features that are products of at most degree distinct input features.
                  </p>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-muted"
                      checked={config.include_bias || false}
                      onChange={(e) => updateConfig({ include_bias: e.target.checked })}
                    />
                    <span className="text-xs">Include Bias</span>
                  </label>
                  
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      className="rounded border-muted"
                      checked={config.include_input_features || false}
                      onChange={(e) => updateConfig({ include_input_features: e.target.checked })}
                    />
                    <span className="text-xs">Include Input Features</span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  },

  validate: (data) => {
    if (!data.columns || data.columns.length === 0) {
      return { isValid: false, message: 'Select at least one input column.' };
    }
    if (data.degree < 2) {
      return { isValid: false, message: 'Degree must be at least 2.' };
    }
    return { isValid: true };
  }
};
