import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Replace, Plus, Trash2, Search, ArrowRight, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

// --- Types ---

interface ReplacementItem {
  old: any;
  new: any;
  oldType: 'string' | 'number' | 'boolean' | 'null' | 'nan';
  newType: 'string' | 'number' | 'boolean' | 'null' | 'nan';
}

interface ValueReplacementConfig {
  columns: string[];
  replacements: ReplacementItem[];
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
          onChange={e => setSearch(e.target.value)}
        />
      </div>
      <div className="overflow-y-auto p-1 space-y-0.5">
        {filtered.length > 0 ? (
          filtered.map(col => {
            const isSelected = selected.includes(col);
            return (
              <div
                key={col}
                onClick={() => toggle(col)}
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

const TypedInput: React.FC<{
  value: any;
  type: string;
  onChange: (val: any, type: any) => void;
  placeholder?: string;
}> = ({ value, type, onChange, placeholder }) => {
  
  const handleTypeChange = (newType: string) => {
    let newValue = value;
    if (newType === 'number') newValue = Number(value) || 0;
    if (newType === 'boolean') newValue = Boolean(value);
    if (newType === 'string') newValue = String(value);
    if (newType === 'null') newValue = null;
    if (newType === 'nan') newValue = 'NaN'; // We'll handle NaN as a special string or null in backend? 
    // Actually JSON doesn't support NaN. We might need to send null or a special string.
    // For now let's assume backend handles None as NaN for numeric columns?
    // Or we can use a special string token if needed.
    // But standard JSON serialization of NaN is null.
    
    onChange(newValue, newType);
  };

  const handleValueChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    let val: any = e.target.value;
    if (type === 'number') val = parseFloat(val);
    onChange(val, type);
  };

  return (
    <div className="flex gap-1">
      <select
        className="w-20 text-[10px] rounded border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 px-1"
        value={type}
        onChange={(e) => handleTypeChange(e.target.value)}
      >
        <option value="string">Text</option>
        <option value="number">Num</option>
        <option value="boolean">Bool</option>
        <option value="null">Null</option>
      </select>
      
      {type === 'string' && (
        <input
          type="text"
          className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1"
          value={value || ''}
          onChange={handleValueChange}
          placeholder={placeholder}
        />
      )}
      {type === 'number' && (
        <input
          type="number"
          className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1"
          value={value}
          onChange={handleValueChange}
          placeholder={placeholder}
        />
      )}
      {type === 'boolean' && (
        <select
          className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1"
          value={String(value)}
          onChange={(e) => onChange(e.target.value === 'true', 'boolean')}
        >
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      )}
      {(type === 'null' || type === 'nan') && (
        <div className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1 bg-gray-100 dark:bg-gray-800 text-gray-500 italic flex items-center">
          {type === 'null' ? 'None / Null' : 'NaN'}
        </div>
      )}
    </div>
  );
};

const ReplacementEditor: React.FC<{
  item: ReplacementItem;
  onChange: (item: ReplacementItem) => void;
  onDelete: () => void;
}> = ({ item, onChange, onDelete }) => {
  return (
    <div className="p-2 border rounded-md bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Find</div>
        <button onClick={onDelete} className="text-gray-400 hover:text-red-500 transition-colors p-1">
          <Trash2 size={14} />
        </button>
      </div>
      
      <TypedInput
        value={item.old}
        type={item.oldType}
        onChange={(val, type) => onChange({ ...item, old: val, oldType: type })}
        placeholder="Value to find"
      />
      
      <div className="flex items-center gap-2">
        <div className="text-gray-400">
          <ArrowRight size={14} className="rotate-90" />
        </div>
        <div className="text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Replace With</div>
      </div>

      <TypedInput
        value={item.new}
        type={item.newType}
        onChange={(val, type) => onChange({ ...item, new: val, newType: type })}
        placeholder="New value"
      />
    </div>
  );
};

const ValueReplacementSettings: React.FC<{ config: ValueReplacementConfig; onChange: (c: ValueReplacementConfig) => void; nodeId?: string }> = ({
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
  
  const columns = schema 
    ? Object.values(schema.columns).map((c: any) => c.name)
    : [];

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > 400);
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const addReplacement = () => {
    onChange({
      ...config,
      replacements: [...config.replacements, { old: '', new: '', oldType: 'string', newType: 'string' }]
    });
  };

  const updateReplacement = (index: number, item: ReplacementItem) => {
    const newItems = [...config.replacements];
    newItems[index] = item;
    onChange({ ...config, replacements: newItems });
  };

  const removeReplacement = (index: number) => {
    const newItems = config.replacements.filter((_, i) => i !== index);
    onChange({ ...config, replacements: newItems });
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
              columns={columns} 
              selected={config.columns} 
              onChange={(cols) => onChange({ ...config, columns: cols })} 
            />
          </div>
        </div>

        {/* Right Column: Replacements */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
            <button 
              onClick={() => setShowInfo(!showInfo)}
              className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
              <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
                About Value Replacement
              </span>
              {showInfo ? (
                <ChevronUp size={14} className="text-blue-600 dark:text-blue-400" />
              ) : (
                <ChevronDown size={14} className="text-blue-600 dark:text-blue-400" />
              )}
            </button>
            
            {showInfo && (
              <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
                <p>Replace specific values with new ones. Useful for fixing typos, recoding categories (e.g., "Yes" to 1), or handling specific placeholders.</p>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Value Mappings
            </label>
            <button
              onClick={addReplacement}
              className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            >
              <Plus size={14} />
              Add Mapping
            </button>
          </div>

          <div className="space-y-3">
            {config.replacements.length === 0 ? (
              <div className="text-center p-6 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-lg">
                <p className="text-xs text-gray-500">No replacements defined.</p>
                <button
                  onClick={addReplacement}
                  className="mt-2 text-xs text-blue-600 hover:underline"
                >
                  Add your first replacement
                </button>
              </div>
            ) : (
              config.replacements.map((item, idx) => (
                <ReplacementEditor
                  key={idx}
                  item={item}
                  onChange={(newItem) => updateReplacement(idx, newItem)}
                  onDelete={() => removeReplacement(idx)}
                />
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export const ValueReplacementNode: NodeDefinition<ValueReplacementConfig> = {
  type: 'ValueReplacement',
  label: 'Value Replacement',
  category: 'Preprocessing',
  description: 'Replace specific values in columns.',
  icon: Replace,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Processed Data', type: 'dataset' }],
  settings: ValueReplacementSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column.' };
    if (config.replacements.length === 0) return { isValid: false, message: 'Add at least one replacement.' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    replacements: [{ old: '', new: '', oldType: 'string', newType: 'string' }],
  }),
};
