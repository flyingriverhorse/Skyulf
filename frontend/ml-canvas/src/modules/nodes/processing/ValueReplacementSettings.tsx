import React, { useState } from 'react';
import { Plus, Trash2, Search, ArrowRight } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

export interface ReplacementItem {
  old: unknown;
  new: unknown;
  oldType: 'string' | 'number' | 'boolean' | 'null' | 'nan';
  newType: 'string' | 'number' | 'boolean' | 'null' | 'nan';
}

export interface ValueReplacementConfig {
  columns: string[];
  replacements: ReplacementItem[];
}

type ValueReplacementValueType = ReplacementItem['oldType'];

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

const TypedInput: React.FC<{
  value: unknown;
  type: ValueReplacementValueType;
  onChange: (val: unknown, type: ValueReplacementValueType) => void;
  placeholder?: string;
}> = ({ value, type, onChange, placeholder }) => {
  
  const handleTypeChange = (newType: ValueReplacementValueType) => {
    let newValue = value;
    if (newType === 'number') newValue = Number(value) || 0;
    if (newType === 'boolean') newValue = Boolean(value);
    if (newType === 'string') newValue = String(value);
    if (newType === 'null') newValue = null;
    if (newType === 'nan') newValue = 'NaN'; 
    
    onChange(newValue, newType);
  };

  const handleValueChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    let val: unknown = e.target.value;
    if (type === 'number') val = parseFloat(val as string);
    onChange(val, type);
  };

  return (
    <div className="flex gap-1">
      <select
        className="w-20 text-[10px] rounded border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 px-1"
        value={type}
        onChange={(e) => { handleTypeChange(e.target.value as ValueReplacementValueType); }}
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
          value={(value as string) || ''}
          onChange={handleValueChange}
          placeholder={placeholder}
        />
      )}
      {type === 'number' && (
        <input
          type="number"
          className="flex-1 min-w-0 text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1"
          value={(value as number) || ''}
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

export const ValueReplacementSettings: React.FC<{ config: ValueReplacementConfig; onChange: (c: ValueReplacementConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find((d: unknown) => (d as Record<string, unknown>)?.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns).map((c) => c.name) : [];

  const addReplacement = () => {
    onChange({
      ...config,
      replacements: [
        ...config.replacements,
        { old: '', new: '', oldType: 'string', newType: 'string' }
      ]
    });
  };

  const updateReplacement = (index: number, item: ReplacementItem) => {
    const newReplacements = [...config.replacements];
    newReplacements[index] = item;
    onChange({ ...config, replacements: newReplacements });
  };

  const removeReplacement = (index: number) => {
    const newReplacements = config.replacements.filter((_, i) => i !== index);
    onChange({ ...config, replacements: newReplacements });
  };

  return (
    <div className="space-y-4 p-1">
      <div className="space-y-2">
        <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Target Columns</label>
        <ColumnSelector
          columns={availableColumns}
          selected={config.columns}
          onChange={(cols) => onChange({ ...config, columns: cols })}
        />
        <p className="text-[10px] text-gray-500">
          Select columns to apply replacements to. If empty, applies to all compatible columns.
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Replacements</label>
          <button
            onClick={addReplacement}
            className="text-xs flex items-center gap-1 text-blue-600 hover:text-blue-700 dark:text-blue-400"
          >
            <Plus size={12} /> Add
          </button>
        </div>
        
        <div className="space-y-3">
          {config.replacements.map((item, i) => (
            <ReplacementEditor
              key={i}
              item={item}
              onChange={(newItem) => { updateReplacement(i, newItem); }}
              onDelete={() => { removeReplacement(i); }}
            />
          ))}
          {config.replacements.length === 0 && (
            <div className="text-center py-4 text-xs text-gray-500 border border-dashed rounded">
              No replacements defined.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
