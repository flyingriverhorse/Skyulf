import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Eraser, Plus, Trash2, Search, Wand2, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

// --- Types ---

interface TextOperation {
  op: 'trim' | 'case' | 'remove_special' | 'regex';
  mode: string;
  replacement?: string; // For remove_special
  pattern?: string; // For regex custom
  repl?: string; // For regex custom
}

interface TextCleaningConfig {
  columns: string[];
  operations: TextOperation[];
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

const OperationEditor: React.FC<{
  operation: TextOperation;
  onChange: (op: TextOperation) => void;
  onDelete: () => void;
}> = ({ operation, onChange, onDelete }) => {
  return (
    <div className="p-3 border rounded-md bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 space-y-3">
      <div className="flex items-center justify-between">
        <select
          className="text-xs font-medium bg-transparent border-none focus:ring-0 p-0 text-gray-900 dark:text-gray-100"
          value={operation.op}
          onChange={(e) => {
            const newOp = e.target.value as any;
            let defaultMode = 'both';
            if (newOp === 'case') defaultMode = 'lower';
            if (newOp === 'remove_special') defaultMode = 'keep_alphanumeric';
            if (newOp === 'regex') defaultMode = 'custom';
            onChange({ ...operation, op: newOp, mode: defaultMode });
          }}
        >
          <option value="trim">Trim Whitespace</option>
          <option value="case">Change Case</option>
          <option value="remove_special">Remove Special Chars</option>
          <option value="regex">Regex Replace</option>
        </select>
        <button onClick={onDelete} className="text-gray-400 hover:text-red-500 transition-colors">
          <Trash2 size={14} />
        </button>
      </div>

      <div className="space-y-2">
        {operation.op === 'trim' && (
          <select
            className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
            value={operation.mode}
            onChange={(e) => onChange({ ...operation, mode: e.target.value })}
          >
            <option value="both">Both Ends</option>
            <option value="leading">Leading Only</option>
            <option value="trailing">Trailing Only</option>
          </select>
        )}

        {operation.op === 'case' && (
          <select
            className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
            value={operation.mode}
            onChange={(e) => onChange({ ...operation, mode: e.target.value })}
          >
            <option value="lower">Lower Case</option>
            <option value="upper">Upper Case</option>
            <option value="title">Title Case</option>
            <option value="sentence">Sentence Case</option>
          </select>
        )}

        {operation.op === 'remove_special' && (
          <div className="space-y-2">
            <select
              className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
              value={operation.mode}
              onChange={(e) => onChange({ ...operation, mode: e.target.value })}
            >
              <option value="keep_alphanumeric">Keep Alphanumeric</option>
              <option value="keep_alphanumeric_space">Keep Alphanumeric & Space</option>
              <option value="letters_only">Letters Only</option>
              <option value="digits_only">Digits Only</option>
            </select>
            <input
              type="text"
              className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
              placeholder="Replacement (optional)"
              value={operation.replacement || ''}
              onChange={(e) => onChange({ ...operation, replacement: e.target.value })}
            />
          </div>
        )}

        {operation.op === 'regex' && (
          <div className="space-y-2">
            <select
              className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
              value={operation.mode}
              onChange={(e) => onChange({ ...operation, mode: e.target.value })}
            >
              <option value="custom">Custom Pattern</option>
              <option value="collapse_whitespace">Collapse Whitespace</option>
              <option value="extract_digits">Extract Digits</option>
              <option value="normalize_slash_dates">Normalize Dates (MM/DD/YYYY)</option>
            </select>
            {operation.mode === 'custom' && (
              <>
                <div className="flex gap-1">
                  <input
                    type="text"
                    className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5 font-mono"
                    placeholder="Regex Pattern"
                    value={operation.pattern || ''}
                    onChange={(e) => onChange({ ...operation, pattern: e.target.value })}
                  />
                  <button
                    className="p-1.5 bg-gray-100 dark:bg-gray-700 rounded border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                    title="Generate regex from literal text"
                    onClick={() => {
                      const text = prompt("Enter the exact text you want to match (it will be escaped):");
                      if (text) {
                        // Simple regex escape
                        const escaped = text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                        onChange({ ...operation, pattern: escaped });
                      }
                    }}
                  >
                    <Wand2 size={14} />
                  </button>
                </div>
                <input
                  type="text"
                  className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-1.5"
                  placeholder="Replacement"
                  value={operation.repl || ''}
                  onChange={(e) => onChange({ ...operation, repl: e.target.value })}
                />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const TextCleaningSettings: React.FC<{ config: TextCleaningConfig; onChange: (c: TextCleaningConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  const [showInfo, setShowInfo] = useState(true);

  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find((d: any) => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  
  // Filter for text columns only
  const textColumns = schema 
    ? Object.values(schema.columns)
        .filter((c: any) => c.dtype === 'object' || c.dtype === 'string' || c.dtype === 'category')
        .map((c: any) => c.name)
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

  const addOperation = () => {
    onChange({
      ...config,
      operations: [...config.operations, { op: 'trim', mode: 'both' }]
    });
  };

  const updateOperation = (index: number, op: TextOperation) => {
    const newOps = [...config.operations];
    newOps[index] = op;
    onChange({ ...config, operations: newOps });
  };

  const removeOperation = (index: number) => {
    const newOps = config.operations.filter((_, i) => i !== index);
    onChange({ ...config, operations: newOps });
  };

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
              columns={textColumns} 
              selected={config.columns} 
              onChange={(cols) => onChange({ ...config, columns: cols })} 
            />
            <p className="text-xs text-gray-500 mt-1">Only text/categorical columns are shown.</p>
          </div>
        </div>

        {/* Right Column: Operations */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
            <button 
              onClick={() => { setShowInfo(!showInfo); }}
              className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
              <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
                About Text Cleaning
              </span>
              {showInfo ? (
                <ChevronUp size={14} className="text-blue-600 dark:text-blue-400" />
              ) : (
                <ChevronDown size={14} className="text-blue-600 dark:text-blue-400" />
              )}
            </button>
            
            {showInfo && (
              <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
                <p>Clean and standardize text columns by trimming whitespace, changing case, removing special characters, or applying custom regex patterns.</p>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Cleaning Operations
            </label>
            <button
              onClick={addOperation}
              className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            >
              <Plus size={14} />
              Add Step
            </button>
          </div>

          <div className="space-y-3">
            {config.operations.length === 0 ? (
              <div className="text-center p-6 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg">
                <p className="text-xs text-gray-500">No operations added.</p>
                <button
                  onClick={addOperation}
                  className="mt-2 text-xs text-blue-600 hover:underline"
                >
                  Add your first cleaning step
                </button>
              </div>
            ) : (
              config.operations.map((op, idx) => (
                <OperationEditor
                  key={idx}
                  operation={op}
                  onChange={(newOp) => { updateOperation(idx, newOp); }}
                  onDelete={() => { removeOperation(idx); }}
                />
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export const TextCleaningNode: NodeDefinition<TextCleaningConfig> = {
  type: 'TextCleaning',
  label: 'Text Cleaning',
  category: 'Preprocessing',
  description: 'Clean and standardize text columns.',
  icon: Eraser,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: TextCleaningSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column.' };
    if (config.operations.length === 0) return { isValid: false, message: 'Add at least one operation.' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    operations: [{ op: 'trim', mode: 'both' }],
  }),
};
