import React, { useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Calculator, Trash2, Calendar, Percent, GitCompare, Search, Check, ChevronDown, ChevronRight, Zap, FunctionSquare, Info, Lightbulb, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { Recommendation } from '../../../core/api/client';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface MathOperation {
  operation_type: 'arithmetic' | 'datetime_extract' | 'ratio' | 'similarity' | 'polynomial' | 'group_agg';
  method: string;
  input_columns: string[];
  secondary_columns?: string[]; // For arithmetic (second operand)
  constants?: number[];
  output_column?: string;
  datetime_features?: string[]; // For datetime_extract
  
  // Polynomial specific
  degree?: number;
  interaction_only?: boolean;
  include_bias?: boolean;

  isExpanded?: boolean; // UI state
}

interface FeatureGenerationConfig {
  operations: MathOperation[];
}

const OPERATION_TYPES = [
  { value: 'arithmetic', label: 'Arithmetic', icon: Calculator },
  { value: 'datetime_extract', label: 'Date Extraction', icon: Calendar },
  { value: 'ratio', label: 'Ratio', icon: Percent },
  { value: 'similarity', label: 'Similarity', icon: GitCompare },
  { value: 'polynomial', label: 'Polynomial', icon: Zap },
  { value: 'group_agg', label: 'Group Agg', icon: FunctionSquare },
];

const ARITHMETIC_METHODS = ['add', 'subtract', 'multiply', 'divide'];
const DATE_METHODS = ['year', 'month', 'day', 'weekday', 'hour', 'quarter', 'is_weekend'];
const SIMILARITY_METHODS = ['ratio', 'token_sort_ratio', 'token_set_ratio'];
const GROUP_AGG_METHODS = ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'];

// --- Helper Components ---

const ColumnSelector: React.FC<{
  columns: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  label?: string;
  single?: boolean;
}> = ({ columns, selected, onChange, label, single }) => {
  const [search, setSearch] = useState('');
  
  const filtered = columns.filter(c => c.toLowerCase().includes(search.toLowerCase()));

  const toggle = (col: string) => {
    if (single) {
      onChange([col]);
    } else {
      if (selected.includes(col)) {
        onChange(selected.filter(c => c !== col));
      } else {
        onChange([...selected, col]);
      }
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
                    {isSelected && <Check size={8} strokeWidth={4} />}
                  </div>
                  <span className="truncate">{col}</span>
                </div>
              );
            })
          ) : (
            <div className="p-2 text-xs text-muted-foreground text-center italic">No columns found</div>
          )}
        </div>
      </div>
    </div>
  );
};

const FeatureGenerationSettings: React.FC<{ config: FeatureGenerationConfig; onChange: (c: FeatureGenerationConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = nodeResult?.metrics;

  const [showRecommendations, setShowRecommendations] = useState(false);



  const allColumns = schema ? Object.values(schema.columns).map(c => c.name) : [];
  const numericColumns = schema 
    ? Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype?.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];
  const dateColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['date', 'time'].some(t => c.dtype?.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];
  const stringColumns = schema
    ? Object.values(schema.columns)
        .filter(c => ['string', 'object', 'text'].some(t => c.dtype?.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];

  // Recommendations
  const recommendations = useRecommendations(nodeId || '', {
    types: ['feature_generation'],
    suggestedNodeTypes: ['FeatureGenerationNode'],
    scope: 'column'
  });

  const handleApplyRecommendation = (rec: Recommendation) => {
    console.log("Applying recommendation:", rec);
  };

  const addOperation = (type: MathOperation['operation_type']) => {
    const newOp: MathOperation = {
      operation_type: type,
      method: type === 'arithmetic' ? 'add' : 
              type === 'similarity' ? 'ratio' : 
              type === 'group_agg' ? 'mean' :
              type === 'polynomial' ? 'poly' : 'year',
      input_columns: [],
      output_column: '',
      datetime_features: type === 'datetime_extract' ? ['year'] : undefined,
      degree: type === 'polynomial' ? 2 : undefined,
      interaction_only: type === 'polynomial' ? false : undefined,
      isExpanded: true
    };
    onChange({ operations: [...(config.operations || []), newOp] });
  };

  const updateOperation = (index: number, updates: Partial<MathOperation>) => {
    const newOps = [...(config.operations || [])];
    newOps[index] = { ...newOps[index], ...updates };
    onChange({ operations: newOps });
  };

  const removeOperation = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const newOps = [...(config.operations || [])];
    newOps.splice(index, 1);
    onChange({ operations: newOps });
  };

  const toggleExpand = (index: number) => {
    const newOps = [...(config.operations || [])];
    newOps[index] = { ...newOps[index], isExpanded: !newOps[index].isExpanded };
    onChange({ operations: newOps });
  };

  const generateDefaultName = (op: MathOperation, index: number) => {
    return `${op.operation_type}_${index}`;
  };

  return (
    <div className="flex h-full">
      {/* Left Sidebar: Tools */}
      <div className="w-20 border-r bg-muted/10 flex flex-col items-center py-4 gap-2 overflow-y-auto shrink-0">
        <span className="text-[10px] font-bold text-muted-foreground mb-2 uppercase tracking-wider">Add</span>
        {OPERATION_TYPES.map(t => (
          <button
            key={t.value}
            onClick={() => { addOperation(t.value as any); }}
            className="flex flex-col items-center justify-center gap-1 p-1.5 rounded-md border bg-card hover:bg-accent hover:border-primary/50 transition-all group w-16 h-14 shadow-sm"
            title={t.label}
          >
            <t.icon size={18} className="text-muted-foreground group-hover:text-primary" />
            <span className="text-[9px] font-medium text-muted-foreground group-hover:text-foreground text-center leading-tight line-clamp-2">
              {t.label.replace(' ', '\n')}
            </span>
          </button>
        ))}
      </div>

      {/* Right Main: Content */}
      <div className="flex-1 flex flex-col min-w-0 bg-background">
      {/* Recommendations Toggle */}
      {recommendations.length > 0 && (
        <div className="border-b bg-muted/10">
          <button 
            onClick={() => { setShowRecommendations(!showRecommendations); }}
            className="w-full flex items-center justify-between px-4 py-2 text-xs font-medium text-muted-foreground hover:text-primary hover:bg-muted/20 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Lightbulb size={14} className={recommendations.length > 0 ? "text-yellow-500" : ""} />
              <span>Recommendations ({recommendations.length})</span>
            </div>
            {showRecommendations ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
          
          {showRecommendations && (
            <div className="p-4 bg-muted/5 border-t">
              <RecommendationsPanel
                recommendations={recommendations}
                onApply={handleApplyRecommendation}
                className="mb-0"
              />
            </div>
          )}
        </div>
      )}



      {/* Main Content Area - Scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {(config.operations || []).map((op, idx) => (
          <div key={idx} className="border rounded-lg bg-card shadow-sm overflow-hidden">
            {/* Header */}
            <div 
              className="flex items-center justify-between px-3 py-2 bg-muted/30 border-b cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => { toggleExpand(idx); }}
            >
              <div className="flex items-center gap-2">
                {op.isExpanded ? <ChevronDown size={14} className="text-muted-foreground" /> : <ChevronRight size={14} className="text-muted-foreground" />}
                <span className="text-[10px] font-bold uppercase tracking-wider text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                  {op.operation_type === 'datetime_extract' ? 'Date' : op.operation_type.replace('_', ' ')}
                </span>
                {['arithmetic', 'similarity', 'group_agg'].includes(op.operation_type) && (
                  <select
                    className="text-sm border-none bg-transparent font-semibold focus:ring-0 cursor-pointer hover:text-primary"
                    value={op.method}
                    onClick={(e) => { e.stopPropagation(); }}
                    onChange={(e) => { updateOperation(idx, { method: e.target.value }); }}
                  >
                    {(op.operation_type === 'arithmetic' ? ARITHMETIC_METHODS : 
                      op.operation_type === 'similarity' ? SIMILARITY_METHODS :
                      op.operation_type === 'group_agg' ? GROUP_AGG_METHODS :
                      []).map(m => (
                      <option key={m} value={m}>{m.replace('_', ' ')}</option>
                    ))}
                  </select>
                )}
              </div>
              <button
                onClick={(e) => { removeOperation(idx, e); }}
                className="text-muted-foreground hover:text-destructive transition-colors p-1 rounded-full hover:bg-destructive/10"
              >
                <Trash2 size={14} />
              </button>
            </div>

            {/* Body */}
            {op.isExpanded && (
            <div className="p-3 space-y-4">
              


              {/* ARITHMETIC */}
              {op.operation_type === 'arithmetic' && (
                <div className="flex flex-col gap-3">
                  <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
                    {op.method === 'divide' 
                      ? 'Performs row-by-row division (Col A / Col B) for each record.' 
                      : 'Performs row-by-row arithmetic between two columns.'}
                  </div>
                  <ColumnSelector
                    label="Column A (Left Operand)"
                    columns={numericColumns}
                    selected={op.input_columns.slice(0, 1)}
                    onChange={(cols) => { updateOperation(idx, { input_columns: cols }); }}
                    single
                  />
                  
                  <div className="flex items-center gap-2">
                     <div className="h-px bg-border flex-1"></div>
                     <span className="text-lg font-bold text-muted-foreground bg-muted/20 w-8 h-8 rounded flex items-center justify-center">
                      {op.method === 'add' ? '+' : op.method === 'subtract' ? '-' : op.method === 'multiply' ? '×' : '÷'}
                    </span>
                     <div className="h-px bg-border flex-1"></div>
                  </div>

                  <ColumnSelector
                    label="Column B (Right Operand)"
                    columns={numericColumns}
                    selected={op.secondary_columns?.slice(0, 1) || []}
                    onChange={(cols) => { updateOperation(idx, { secondary_columns: cols }); }}
                    single
                  />
                </div>
              )}

              {/* RATIO */}
              {op.operation_type === 'ratio' && (
                <div className="space-y-3">
                  <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
                    Calculates aggregate ratio: <strong>Sum(Numerator) / Sum(Denominator)</strong>.
                  </div>
                  <ColumnSelector
                    label="Numerator (Sum)"
                    columns={numericColumns}
                    selected={op.input_columns}
                    onChange={(cols) => updateOperation(idx, { input_columns: cols })}
                  />
                  
                  <div className="relative flex items-center justify-center">
                    <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-dashed"></div></div>
                    <span className="relative bg-card px-2 text-xs text-muted-foreground font-medium">Divided By</span>
                  </div>

                  <ColumnSelector
                    label="Denominator (Sum)"
                    columns={numericColumns}
                    selected={op.secondary_columns || []}
                    onChange={(cols) => updateOperation(idx, { secondary_columns: cols })}
                  />
                </div>
              )}

              {/* SIMILARITY */}
              {op.operation_type === 'similarity' && (
                <div className="flex flex-col gap-3">
                  <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20 space-y-1">
                    <p>Calculates string similarity score (0-100) between two text columns.</p>
                    <ul className="list-disc pl-3 space-y-0.5 opacity-80">
                      <li><strong>Ratio:</strong> Strict character matching (Levenshtein). "apple banana" != "banana apple".</li>
                      <li><strong>Token Sort:</strong> Sorts words alphabetically then compares. "apple banana" == "banana apple".</li>
                      <li><strong>Token Set:</strong> Compares intersection of words. Handles duplicates/subsets well.</li>
                    </ul>
                  </div>
                  <ColumnSelector
                    label="String A"
                    columns={stringColumns}
                    selected={op.input_columns.slice(0, 1)}
                    onChange={(cols) => updateOperation(idx, { input_columns: cols })}
                    single
                  />
                  
                  <div className="flex items-center gap-2">
                     <div className="h-px bg-border flex-1"></div>
                     <span className="text-xs font-bold text-muted-foreground bg-muted/20 px-2 py-1 rounded">vs</span>
                     <div className="h-px bg-border flex-1"></div>
                  </div>

                  <ColumnSelector
                    label="String B"
                    columns={stringColumns}
                    selected={op.secondary_columns?.slice(0, 1) || []}
                    onChange={(cols) => updateOperation(idx, { secondary_columns: cols })}
                    single
                  />
                </div>
              )}

              {/* POLYNOMIAL */}
              {op.operation_type === 'polynomial' && (
                <div className="space-y-3">
                  <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
                    Generates polynomial and interaction features (e.g., A², A×B).
                  </div>
                  <ColumnSelector
                    label="Input Columns"
                    columns={numericColumns}
                    selected={op.input_columns}
                    onChange={(cols) => updateOperation(idx, { input_columns: cols })}
                  />
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label className="text-xs font-medium text-muted-foreground">Degree</label>
                      <input
                        type="number"
                        min="2"
                        max="5"
                        className="w-full text-xs border rounded p-1.5"
                        value={op.degree || 2}
                        onChange={(e) => { updateOperation(idx, { degree: parseInt(e.target.value) }); }}
                      />
                    </div>
                    <div className="space-y-1 pt-5">
                      <label className="flex items-center gap-2 text-xs cursor-pointer">
                        <input
                          type="checkbox"
                          checked={op.interaction_only || false}
                          onChange={(e) => { updateOperation(idx, { interaction_only: e.target.checked }); }}
                        />
                        Interaction Only
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* GROUP AGG */}
              {op.operation_type === 'group_agg' && (
                <div className="space-y-3">
                  <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
                    Calculates aggregate statistics (e.g., Mean Salary) grouped by a categorical column (e.g., Department) and assigns it back to each row.
                  </div>
                  <ColumnSelector
                    label="Group By (Categorical)"
                    columns={stringColumns.length > 0 ? stringColumns : allColumns}
                    selected={op.input_columns.slice(0, 1)}
                    onChange={(cols) => updateOperation(idx, { input_columns: cols })}
                    single
                  />
                  
                  <div className="relative flex items-center justify-center">
                    <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-dashed"></div></div>
                    <span className="relative bg-card px-2 text-xs text-muted-foreground font-medium">Target Column</span>
                  </div>

                  <ColumnSelector
                    label="Target (Numeric)"
                    columns={numericColumns}
                    selected={op.secondary_columns?.slice(0, 1) || []}
                    onChange={(cols) => updateOperation(idx, { secondary_columns: cols })}
                    single
                  />
                </div>
              )}

              {/* DATE EXTRACT */}
              {op.operation_type === 'datetime_extract' && (
                <div className="space-y-3">
                  <ColumnSelector
                    label="Date Column"
                    columns={dateColumns.length > 0 ? dateColumns : allColumns}
                    selected={op.input_columns.slice(0, 1)}
                    onChange={(cols) => updateOperation(idx, { input_columns: cols })}
                    single
                  />
                  
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-muted-foreground">Features to Extract</label>
                    <div className="grid grid-cols-2 gap-2">
                      {DATE_METHODS.map(method => (
                        <label key={method} className="flex items-center gap-2 text-xs p-1.5 border rounded hover:bg-accent cursor-pointer transition-colors">
                          <input
                            type="checkbox"
                            className="rounded border-muted-foreground/40 text-primary focus:ring-0"
                            checked={(op.datetime_features || []).includes(method)}
                            onChange={(e) => {
                              const current = op.datetime_features || [];
                              const newFeatures = e.target.checked
                                ? [...current, method]
                                : current.filter(f => f !== method);
                              updateOperation(idx, { datetime_features: newFeatures });
                            }}
                          />
                          <span className="capitalize">{method.replace('_', ' ')}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Output Name */}
              <div className="pt-2">
                <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1 block">Output Column Name</label>
                <input
                  type="text"
                  className="w-full text-xs border rounded px-2 py-1.5 bg-background focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all"
                  placeholder={generateDefaultName(op, idx)}
                  value={op.output_column || ''}
                  onChange={(e) => { updateOperation(idx, { output_column: e.target.value }); }}
                />
              </div>
            </div>
            )}
          </div>
        ))}

        {(!config.operations || config.operations.length === 0) && (
          <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed rounded-lg bg-muted/5 text-muted-foreground">
            <Calculator size={32} className="mb-2 opacity-20" />
            <p className="text-xs font-medium">No operations added</p>
            <p className="text-[10px] opacity-70">Select a type above to start</p>
          </div>
        )}

        {/* Feedback Section */}
        {metrics && (
          <div className="border rounded-lg bg-muted/10 p-4">
            <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-primary">
              <Activity size={14} />
              <span>Last Run Results</span>
            </div>
            
            <div className="space-y-2 text-xs">
              {metrics.generated_features && metrics.generated_features.length > 0 ? (
                <>
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground">New Features Created:</span>
                    <span className="font-medium text-primary bg-primary/10 px-2 py-0.5 rounded-full">
                      {metrics.generated_features.length}
                    </span>
                  </div>
                  
                  <div className="pt-2">
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold mb-1.5 block">Generated Columns</span>
                    <div className="flex flex-wrap gap-1.5">
                      {metrics.generated_features.map((f: string) => (
                        <span key={f} className="px-2 py-1 bg-background border rounded text-[10px] font-mono text-foreground shadow-sm">
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex items-center gap-2 text-muted-foreground italic">
                  <Info size={12} />
                  <span>No features generated in last run.</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  );
};

export const FeatureGenerationNode: NodeDefinition = {
  type: 'FeatureGenerationNode',
  label: 'Feature Generation',
  description: 'Create new features via math, stats, or date extraction.',
  icon: Calculator,
  category: 'Preprocessing',
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Enhanced' }],
  validate: (config: FeatureGenerationConfig) => {
    if (!config.operations || config.operations.length === 0) {
      return { isValid: false, message: 'Add at least one operation.' };
    }
    for (const op of config.operations) {

      if (op.operation_type === 'arithmetic' && (op.input_columns.length === 0 || (!op.secondary_columns?.length && !op.constants?.length))) {
        return { isValid: false, message: 'Arithmetic requires two operands.' };
      }
      if (op.operation_type === 'datetime_extract' && op.input_columns.length === 0) {
        return { isValid: false, message: 'Select a date column.' };
      }
      if (op.operation_type === 'polynomial' && op.input_columns.length === 0) {
        return { isValid: false, message: 'Select input columns for polynomial features.' };
      }
      if (op.operation_type === 'group_agg' && (op.input_columns.length === 0 || !op.secondary_columns?.length)) {
        return { isValid: false, message: 'Select both Group By and Target columns.' };
      }
    }
    return { isValid: true };
  },
  settings: FeatureGenerationSettings,
  getDefaultConfig: () => ({
    operations: []
  })
};
