import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { ArrowLeftRight, Plus, Trash2, Search, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

// --- Types ---

interface AliasReplacementConfig {
  columns: string[];
  mode: 'custom' | 'canonicalize_country_codes' | 'normalize_boolean' | 'punctuation';
  custom_pairs: Record<string, string>;
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

const CustomPairEditor: React.FC<{
  pairs: Record<string, string>;
  onChange: (pairs: Record<string, string>) => void;
}> = ({ pairs, onChange }) => {
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');

  const addPair = () => {
    if (newKey && newValue) {
      onChange({ ...pairs, [newKey]: newValue });
      setNewKey('');
      setNewValue('');
    }
  };

  const removePair = (key: string) => {
    const newPairs = { ...pairs };
    delete newPairs[key];
    onChange(newPairs);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-2 items-end">
        <div className="flex-1 space-y-1">
          <label className="text-[10px] text-gray-500 uppercase font-semibold">Alias (Old)</label>
          <input
            className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1.5"
            placeholder="e.g. NY"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
          />
        </div>
        <div className="flex-1 space-y-1">
          <label className="text-[10px] text-gray-500 uppercase font-semibold">Canonical (New)</label>
          <input
            className="w-full text-xs rounded border border-gray-300 dark:border-gray-600 px-2 py-1.5"
            placeholder="e.g. New York"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addPair()}
          />
        </div>
        <button
          onClick={addPair}
          disabled={!newKey || !newValue}
          className="p-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Plus size={16} />
        </button>
      </div>

      <div className="space-y-2 max-h-40 overflow-y-auto border rounded-md p-2 bg-gray-50 dark:bg-gray-800/50">
        {Object.entries(pairs).length === 0 ? (
          <div className="text-center text-xs text-gray-500 py-2">No custom aliases defined.</div>
        ) : (
          Object.entries(pairs).map(([key, val]) => (
            <div key={key} className="flex items-center justify-between bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 text-xs">
              <div className="flex items-center gap-2 overflow-hidden">
                <span className="font-medium text-red-600 truncate max-w-[80px]">{key}</span>
                <ArrowLeftRight size={12} className="text-gray-400 shrink-0" />
                <span className="font-medium text-green-600 truncate max-w-[80px]">{val}</span>
              </div>
              <button onClick={() => removePair(key)} className="text-gray-400 hover:text-red-500">
                <Trash2 size={14} />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

const AliasReplacementSettings: React.FC<{ config: AliasReplacementConfig; onChange: (c: AliasReplacementConfig) => void; nodeId?: string }> = ({
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
    return () => observer.disconnect();
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
              columns={textColumns} 
              selected={config.columns} 
              onChange={(cols) => onChange({ ...config, columns: cols })} 
            />
            <p className="text-xs text-gray-500 mt-1">Only text/categorical columns are shown.</p>
          </div>
        </div>

        {/* Right Column: Settings */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-100 dark:border-blue-800 overflow-hidden">
            <button 
              onClick={() => setShowInfo(!showInfo)}
              className="w-full flex items-center gap-2 p-3 text-left hover:bg-blue-100/50 dark:hover:bg-blue-900/30 transition-colors"
            >
              <Info className="text-blue-600 dark:text-blue-400 shrink-0" size={16} />
              <span className="text-xs font-semibold text-blue-800 dark:text-blue-200 flex-1">
                About Alias Replacement
              </span>
              {showInfo ? (
                <ChevronUp size={14} className="text-blue-600 dark:text-blue-400" />
              ) : (
                <ChevronDown size={14} className="text-blue-600 dark:text-blue-400" />
              )}
            </button>
            
            {showInfo && (
              <div className="px-3 pb-3 text-xs text-blue-800 dark:text-blue-200 pl-9">
                <p>Standardize data by mapping multiple variations (aliases) to a single canonical value. Useful for country codes, boolean strings, or custom synonyms.</p>
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
              onChange={(e) => onChange({ ...config, mode: e.target.value as any })}
            >
              <option value="custom">Custom Mapping</option>
              <option value="canonicalize_country_codes">Canonicalize Country Codes</option>
              <option value="normalize_boolean">Normalize Booleans (yes/no -&gt; True/False)</option>
              <option value="punctuation">Remove Punctuation</option>
            </select>
            <p className="text-[10px] text-gray-500 mt-1">
              {config.mode === 'custom' && "Define your own alias mappings below."}
              {config.mode === 'canonicalize_country_codes' && "Standardizes country codes (e.g. 'USA', 'U.S.A.' -> 'US')."}
              {config.mode === 'normalize_boolean' && "Converts 'yes', 'y', '1' to True and 'no', 'n', '0' to False."}
              {config.mode === 'punctuation' && "Removes common punctuation characters from text."}
            </p>
          </div>

          {config.mode === 'custom' && (
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">
                Custom Aliases
              </label>
              <CustomPairEditor
                pairs={config.custom_pairs}
                onChange={(pairs) => onChange({ ...config, custom_pairs: pairs })}
              />
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export const AliasReplacementNode: NodeDefinition<AliasReplacementConfig> = {
  type: 'AliasReplacement',
  label: 'Alias Replacement',
  category: 'Preprocessing',
  description: 'Standardize values by mapping aliases to canonical forms.',
  icon: ArrowLeftRight,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Standardized Data', type: 'dataset' }],
  settings: AliasReplacementSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column.' };
    if (config.mode === 'custom' && Object.keys(config.custom_pairs).length === 0) {
      return { isValid: false, message: 'Add at least one alias pair for custom mode.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    mode: 'custom',
    custom_pairs: {},
  }),
};
