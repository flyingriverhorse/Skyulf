import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Hash, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';

interface EncodingConfig {
  method: 'onehot' | 'ordinal' | 'label' | 'target' | 'hash' | 'dummy';
  columns: string[];
  // OneHot/Dummy specific
  drop_first?: boolean;
  drop_original?: boolean;
  handle_unknown?: 'error' | 'ignore';
  // Hash specific
  n_features?: number;
  // Target specific
  target_column?: string;
  // Label/Ordinal specific
  unknown_value?: number; // For Ordinal
  missing_code?: number; // For Label
}

const EncodingSettings: React.FC<{ config: EncodingConfig; onChange: (c: EncodingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const [searchTerm, setSearchTerm] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = nodeResult?.metrics;
  const recommendations = executionResult?.recommendations || [];

  const filteredRecommendations = recommendations.filter(rec => 
    rec.suggested_node_type === 'encoding' || 
    rec.type.includes('encoding') ||
    rec.type.includes('cardinality')
  );

  // Responsive layout logic
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

  const categoricalColumns = schema 
    ? Object.values(schema.columns)
        .filter(c => ['object', 'string', 'category', 'bool'].some(t => c.dtype?.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];

  const filteredColumns = categoricalColumns.filter(c => 
    c.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSelectAll = () => {
    onChange({ ...config, columns: [...categoricalColumns] });
  };

  const handleDeselectAll = () => {
    onChange({ ...config, columns: [] });
  };

  return (
    <div ref={containerRef} className="p-4 space-y-4">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
          Connect a dataset node to see available columns.
        </div>
      )}
      
      {isLoading && !!datasetId && (
        <div className="text-xs text-muted-foreground animate-pulse">
          Loading schema...
        </div>
      )}

      <div className={`grid gap-4 ${isWide ? 'grid-cols-2' : 'grid-cols-1'}`}>
        {/* Method Selection */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Encoding Method</label>
          <select
            className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
            value={config.method}
            onChange={(e) => onChange({ ...config, method: e.target.value as any })}
          >
            <option value="onehot">One-Hot Encoding</option>
            <option value="dummy">Dummy Encoding</option>
            <option value="label">Label Encoding</option>
            <option value="ordinal">Ordinal Encoding</option>
            <option value="target">Target Encoding</option>
            <option value="hash">Hash Encoding</option>
          </select>
          <p className="text-xs text-muted-foreground">
            {config.method === 'onehot' && "Creates binary columns for each category."}
            {config.method === 'dummy' && "Like One-Hot but drops first category to avoid collinearity."}
            {config.method === 'label' && "Assigns a unique integer to each category."}
            {config.method === 'ordinal' && "Encodes categories as ordered integers."}
            {config.method === 'target' && "Encodes categories based on target mean."}
            {config.method === 'hash' && "Maps categories to a fixed number of features."}
          </p>
          
          <div className="pt-2">
            <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.drop_original !== false} // Default to true
                  onChange={(e) => onChange({ ...config, drop_original: e.target.checked })}
                  className="rounded border-gray-300"
                />
                Drop Original Column
            </label>
            <p className="text-[10px] text-muted-foreground ml-5">
                If unchecked, keeps the original column alongside encoded features.
            </p>
          </div>
        </div>

        {/* Method Specific Settings */}
        <div className="space-y-2">
          {(config.method === 'onehot' || config.method === 'dummy') && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.method === 'dummy' ? true : (config.drop_first || false)}
                  disabled={config.method === 'dummy'}
                  onChange={(e) => onChange({ ...config, drop_first: e.target.checked })}
                  className="rounded border-gray-300"
                />
                Drop First Category
              </label>
              <div className="space-y-1">
                <label className="block text-xs font-medium">Handle Unknown</label>
                <select
                  className="w-full p-1 text-sm border rounded"
                  value={config.handle_unknown || 'ignore'}
                  onChange={(e) => onChange({ ...config, handle_unknown: e.target.value as any })}
                  title="How to handle categories seen in test data but not in training data."
                >
                  <option value="ignore">Ignore (Zeros)</option>
                  <option value="error">Raise Error</option>
                </select>
                <p className="text-[10px] text-muted-foreground">
                  "Ignore" produces all-zeros for unknown categories.
                </p>
              </div>
            </div>
          )}

          {config.method === 'target' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="block text-sm font-medium">Target Column</label>
              <select
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.target_column || ''}
                onChange={(e) => onChange({ ...config, target_column: e.target.value })}
              >
                <option value="">Select a target column...</option>
                {schema && Object.values(schema.columns).map(col => (
                  <option key={col.name} value={col.name}>{col.name} ({col.dtype})</option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">
                The column to use for calculating target statistics.
              </p>
            </div>
          )}

          {config.method === 'hash' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="block text-sm font-medium">Number of Features</label>
              <input
                type="number"
                min="1"
                className="w-full p-2 border rounded"
                value={config.n_features || 8}
                onChange={(e) => onChange({ ...config, n_features: parseInt(e.target.value) })}
              />
            </div>
          )}

          {config.method === 'ordinal' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="block text-sm font-medium">Unknown Value</label>
              <input
                type="number"
                className="w-full p-2 border rounded"
                value={config.unknown_value ?? -1}
                onChange={(e) => onChange({ ...config, unknown_value: parseInt(e.target.value) })}
                title="Integer to assign for unknown categories."
              />
              <p className="text-xs text-muted-foreground">
                Value assigned to unknown categories (default: -1).
              </p>
            </div>
          )}

          {config.method === 'label' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="block text-sm font-medium">Missing/Unknown Code</label>
              <input
                type="number"
                className="w-full p-2 border rounded"
                value={config.missing_code ?? -1}
                onChange={(e) => onChange({ ...config, missing_code: parseInt(e.target.value) })}
                title="Integer to assign for missing or unknown categories."
              />
              <p className="text-xs text-muted-foreground">
                Value assigned to missing/unknown categories (default: -1).
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Column Selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium">Columns to Encode</label>
          <span className="text-xs text-muted-foreground">
            {config.columns.length} selected
          </span>
        </div>

        <div className="space-y-2 mb-2">
          <input 
            type="text" 
            placeholder="Search columns..." 
            className="block w-full px-3 py-1.5 text-sm bg-background border rounded-md shadow-sm focus:ring-1 focus:ring-primary outline-none"
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); }}
          />
          <div className="flex gap-2 text-xs justify-end">
            <button onClick={handleSelectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Select All</button>
            <span className="text-border">|</span>
            <button onClick={handleDeselectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Deselect All</button>
          </div>
        </div>

        {categoricalColumns.length > 0 ? (
          <div className="space-y-1 max-h-40 overflow-y-auto border rounded p-2 bg-background">
            {filteredColumns.map(col => (
              <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-1 rounded cursor-pointer">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 text-primary focus:ring-primary"
                  checked={config.columns.includes(col)}
                  onChange={(e) => {
                    const newCols = e.target.checked
                      ? [...config.columns, col]
                      : config.columns.filter(c => c !== col);
                    onChange({ ...config, columns: newCols });
                  }}
                />
                {col}
              </label>
            ))}
            {filteredColumns.length === 0 && (
              <div className="text-xs text-muted-foreground text-center py-2">
                No columns match "{searchTerm}"
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic border rounded p-4 text-center">
            No categorical columns found
          </div>
        )}
      </div>

      {/* Recommendations Section */}
      {filteredRecommendations.length > 0 && (
        <RecommendationsPanel recommendations={filteredRecommendations} />
      )}

      {/* Feedback Section */}
      {metrics && (
        <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
            <Activity size={14} />
            <span>Last Run Results</span>
          </div>
          <div className="space-y-1 text-xs">
            {metrics.encoded_columns_count !== undefined && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Columns Encoded:</span>
                <span className="font-medium">{metrics.encoded_columns_count}</span>
              </div>
            )}
            {metrics.new_features_count !== undefined && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">New Features Created:</span>
                <span className="font-medium text-primary">{metrics.new_features_count}</span>
              </div>
            )}
            
            {/* Detailed Counts */}
            {metrics.categories_count && (
              <div className="mt-2 border-t pt-2">
                <span className="text-muted-foreground block mb-1 font-medium">Categories Found:</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {Object.entries(metrics.categories_count).map(([col, count]) => (
                    <div key={col} className="flex justify-between text-[10px]">
                      <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                      <span className="font-mono">{count as number}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {metrics.classes_count && (
              <div className="mt-2 border-t pt-2">
                <span className="text-muted-foreground block mb-1 font-medium">Classes Found:</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {Object.entries(metrics.classes_count).map(([col, count]) => (
                    <div key={col} className="flex justify-between text-[10px]">
                      <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                      <span className="font-mono">{count as number}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const EncodingNode: NodeDefinition<EncodingConfig> = {
  type: 'encoding',
  label: 'Encoding',
  category: 'Preprocessing',
  description: 'Encode categorical variables.',
  icon: Hash,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Encoded Data', type: 'dataset' }],
  settings: EncodingSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, error: 'Select at least one column' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'onehot',
    columns: [],
    drop_first: false,
    handle_unknown: 'ignore',
    n_features: 8,
    unknown_value: -1,
    missing_code: -1
  }),
};
