import React, { useMemo, useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { FunctionSquare, Plus, Trash2, Search, ChevronDown, ChevronRight } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';

interface TransformationRule {
  columns: string[];
  method: 'yeo-johnson' | 'box-cox' | 'log' | 'square_root' | 'cube_root' | 'reciprocal' | 'square' | 'exponential';
  params?: {
    standardize?: boolean;
    clip_threshold?: number;
  };
}

interface TransformationConfig {
  transformations: TransformationRule[];
}

const TRANSFORMATION_TYPES = {
  power: {
    label: 'Power Transforms',
    methods: [
      { value: 'yeo-johnson', label: 'Yeo-Johnson' },
      { value: 'box-cox', label: 'Box-Cox (>0 only)' }
    ]
  },
  simple: {
    label: 'Simple Math',
    methods: [
      { value: 'log', label: 'Log (log1p)' },
      { value: 'square_root', label: 'Square Root' },
      { value: 'cube_root', label: 'Cube Root' },
      { value: 'reciprocal', label: 'Reciprocal (1/x)' },
      { value: 'square', label: 'Square (x²)' },
      { value: 'exponential', label: 'Exponential (eˣ)' }
    ]
  }
} as const;

const getMethodType = (method: string): 'power' | 'simple' => {
  if (['yeo-johnson', 'box-cox'].includes(method)) return 'power';
  return 'simple';
};

const ColumnSelector: React.FC<{
  allColumns: string[];
  selectedColumns: string[];
  onChange: (cols: string[]) => void;
}> = ({ allColumns, selectedColumns, onChange }) => {
  const [search, setSearch] = useState("");
  
  const filteredColumns = useMemo(() => 
    allColumns.filter(c => c.toLowerCase().includes(search.toLowerCase())),
  [allColumns, search]);

  const handleSelectAll = () => {
    const newSelection = Array.from(new Set([...selectedColumns, ...filteredColumns]));
    onChange(newSelection);
  };

  const handleDeselectAll = () => {
    const newSelection = selectedColumns.filter(c => !filteredColumns.includes(c));
    onChange(newSelection);
  };

  const toggleColumn = (col: string) => {
    if (selectedColumns.includes(col)) {
      onChange(selectedColumns.filter(c => c !== col));
    } else {
      onChange([...selectedColumns, col]);
    }
  };

  return (
    <div className="space-y-2 border rounded p-2 bg-background">
      <div className="flex gap-2 items-center">
        <div className="relative flex-1">
           <Search className="absolute left-2 top-1.5 w-3 h-3 text-muted-foreground" />
           <input 
             className="w-full pl-7 pr-2 py-1 text-xs border rounded bg-background" 
             placeholder="Search columns..." 
             value={search}
             onChange={e => { setSearch(e.target.value); }}
           />
        </div>
      </div>
      <div className="flex gap-2 text-xs border-b pb-2">
        <button onClick={handleSelectAll} className="text-primary hover:underline font-medium">Select All</button>
        <span className="text-muted-foreground">|</span>
        <button onClick={handleDeselectAll} className="text-muted-foreground hover:underline">None</button>
        <span className="ml-auto text-muted-foreground">{selectedColumns.length} selected</span>
      </div>
      <div className="max-h-32 overflow-y-auto space-y-1 pt-1">
         {filteredColumns.length === 0 ? (
            <div className="text-xs text-muted-foreground p-2 text-center">No columns match "{search}"</div>
         ) : (
            filteredColumns.map(col => (
              <label key={col} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-accent/50 p-1 rounded">
                <input
                  type="checkbox"
                  checked={selectedColumns.includes(col)}
                  onChange={() => { toggleColumn(col); }}
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                />
                <span className="truncate" title={col}>{col}</span>
              </label>
            ))
         )}
      </div>
    </div>
  );
};

const TransformationSettings: React.FC<{ config: TransformationConfig; onChange: (c: TransformationConfig) => void; nodeId?: string }> = ({
  config: propConfig,
  onChange,
  nodeId,
}) => {
  // Defensive defaults
  const config = useMemo(() => propConfig || { transformations: [] }, [propConfig]);
  const transformations = useMemo(() => Array.isArray(config.transformations) ? config.transformations : [], [config.transformations]);

  const [expandedRules, setExpandedRules] = useState<number[]>([]);

  // Responsive Layout Logic
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

  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  
  const columns = useMemo(() => {
    if (!schema?.columns) return [];
    return Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
        .map(c => c.name);
  }, [schema]);

  const addRule = () => {
    const newIndex = transformations.length;
    setExpandedRules(prev => [...prev, newIndex]);
    onChange({
      transformations: [...transformations, { columns: [], method: 'log', params: {} }]
    });
  };

  const updateRule = (index: number, updates: Partial<TransformationRule>) => {
    const current = [...transformations];
    current[index] = { ...current[index], ...updates };
    onChange({ transformations: current });
  };

  const removeRule = (index: number) => {
    const current = [...transformations];
    current.splice(index, 1);
    onChange({ transformations: current });
    setExpandedRules(prev => prev.filter(i => i !== index).map(i => i > index ? i - 1 : i));
  };

  const toggleRule = (index: number) => {
    setExpandedRules(prev => 
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const handleTypeChange = (index: number, newType: 'power' | 'simple') => {
    const defaultMethod = TRANSFORMATION_TYPES[newType].methods[0].value;
    updateRule(index, { method: defaultMethod as TransformationRule['method'] });
  };

  const backendRecommendations = useRecommendations(nodeId || '', {
    types: ['transformation', 'preprocessing'],
    suggestedNodeTypes: ['TransformationNode'],
    scope: 'column'
  });

  return (
    <div ref={containerRef} className="p-4 space-y-6 h-full overflow-y-auto">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium">Transformation Rules</label>
        <button
          onClick={addRule}
          className="flex items-center gap-1 text-xs bg-primary text-primary-foreground px-2 py-1 rounded hover:bg-primary/90"
        >
          <Plus size={12} /> Add Rule
        </button>
      </div>

      <div className="space-y-4">
        {transformations.map((rule, idx) => {
          const currentType = getMethodType(rule.method);
          const isExpanded = expandedRules.includes(idx);
          const methodLabel = TRANSFORMATION_TYPES[currentType].methods.find(m => m.value === rule.method)?.label || rule.method;
          
          return (
            <div key={idx} className="border rounded bg-card shadow-sm overflow-hidden">
              {/* Header */}
              <div 
                className="flex items-center justify-between p-3 cursor-pointer hover:bg-accent/50 transition-colors"
                onClick={() => { toggleRule(idx); }}
              >
                <div className="flex items-center gap-2 overflow-hidden">
                  {isExpanded ? <ChevronDown size={16} className="text-muted-foreground flex-shrink-0" /> : <ChevronRight size={16} className="text-muted-foreground flex-shrink-0" />}
                  <div className="flex flex-col overflow-hidden">
                    <span className="text-sm font-medium truncate">{methodLabel}</span>
                    <span className="text-xs text-muted-foreground truncate">
                      {rule.columns.length === 0 ? 'No columns selected' : `${rule.columns.length} columns`}
                    </span>
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); removeRule(idx); }}
                  className="text-destructive hover:bg-destructive/10 p-1.5 rounded flex-shrink-0"
                  title="Remove rule"
                >
                  <Trash2 size={14} />
                </button>
              </div>

              {/* Body */}
              {isExpanded && (
                <div className={`p-3 border-t bg-accent/5 animate-in slide-in-from-top-2 duration-200 ${isWide ? 'grid grid-cols-2 gap-4' : 'space-y-3'}`}>
                  
                  {/* Left Column: Settings */}
                  <div className="flex flex-col gap-3">
                    <div className="flex flex-col gap-2">
                      <label className="text-xs font-medium text-muted-foreground">Method</label>
                      <select
                        className="w-full p-1.5 border rounded text-xs bg-background font-medium"
                        value={currentType}
                        onChange={(e) => { handleTypeChange(idx, e.target.value as 'power' | 'simple'); }}
                        onClick={(e) => { e.stopPropagation(); }}
                      >
                        <option value="simple">Simple Math</option>
                        <option value="power">Power Transform</option>
                      </select>

                      <select
                        className="w-full p-1.5 border rounded text-xs bg-background"
                        value={rule.method}
                        onChange={(e) => { updateRule(idx, { method: e.target.value as TransformationRule['method'] }); }}
                        onClick={(e) => { e.stopPropagation(); }}
                      >
                        {TRANSFORMATION_TYPES[currentType].methods.map(m => (
                          <option key={m.value} value={m.value}>{m.label}</option>
                        ))}
                      </select>
                    </div>

                    {(rule.method === 'yeo-johnson' || rule.method === 'box-cox') && (
                      <div className="flex items-center gap-2 pt-1" onClick={(e) => { e.stopPropagation(); }}>
                        <input
                          type="checkbox"
                          id={`std-${idx}`}
                          checked={rule.params?.standardize !== false}
                          onChange={(e) => { updateRule(idx, { params: { ...rule.params, standardize: e.target.checked } }); }}
                        />
                        <label htmlFor={`std-${idx}`} className="text-xs text-muted-foreground">Standardize result</label>
                      </div>
                    )}
                    
                    {rule.method === 'exponential' && (
                       <div className="flex flex-col gap-1 pt-1" onClick={(e) => { e.stopPropagation(); }}>
                         <label className="text-xs text-muted-foreground">Clip Threshold:</label>
                         <input
                           type="number"
                           className="w-full p-1 border rounded text-xs"
                           value={rule.params?.clip_threshold || 700}
                           onChange={(e) => { updateRule(idx, { params: { ...rule.params, clip_threshold: parseFloat(e.target.value) } }); }}
                         />
                       </div>
                    )}
                  </div>

                  {/* Right Column: Columns */}
                  <div className="space-y-1 min-w-0" onClick={(e) => { e.stopPropagation(); }}>
                    <label className="text-xs font-medium text-muted-foreground">Target Columns</label>
                    <ColumnSelector 
                      allColumns={columns}
                      selectedColumns={rule.columns}
                      onChange={(newCols) => { updateRule(idx, { columns: newCols }); }}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
        
        {transformations.length === 0 && (
          <div className="text-xs text-muted-foreground text-center py-8 border border-dashed rounded bg-muted/20">
            No transformation rules defined.<br/>Click "Add Rule" to start.
          </div>
        )}
      </div>

      <RecommendationsPanel
        recommendations={backendRecommendations || []}
        onApply={(rec) => {
          // Apply recommendation logic
          const r = rec as unknown as Record<string, unknown>;
          const method = (r.suggested_params as Record<string, unknown>).method || (r.params as Record<string, unknown>)?.method || 'log';
          const columns = (r.target_columns as string[]) || [];
          
          if (columns.length > 0) {
             const newIndex = transformations.length;
             setExpandedRules(prev => [...prev, newIndex]);
             onChange({
               transformations: [
                 ...transformations,
                 { columns, method: method as TransformationRule['method'], params: {} }
               ]
             });
          }
        }}
      />
    </div>
  );
};

export const TransformationNode: NodeDefinition<TransformationConfig> = {
  type: 'TransformationNode',
  label: 'Transformation',
  description: 'Apply mathematical transformations to features.',
  icon: FunctionSquare as unknown as React.FC<any>,
  category: 'Preprocessing',
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Transformed' }],
  validate: (config: TransformationConfig) => {
    if (config.transformations.length === 0) {
      return { isValid: false, message: 'Add at least one transformation rule.' };
    }
    for (const rule of config.transformations) {
        if (rule.columns.length === 0) {
            return { isValid: false, message: 'Each rule must have at least one column selected.' };
        }
    }
    return { isValid: true };
  },
  settings: TransformationSettings,
  getDefaultConfig: () => ({
    transformations: []
  })
};
