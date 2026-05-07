import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Hash, Activity, Info } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';

interface EncodingConfig {
  method: 'onehot' | 'ordinal' | 'label' | 'target' | 'hash' | 'dummy';
  columns: string[];
  // OneHot/Dummy specific
  drop_first?: boolean | undefined;
  drop_original?: boolean | undefined;
  handle_unknown?: 'error' | 'ignore' | 'use_encoded_value' | undefined;
  // OneHot specific
  max_categories?: number | undefined;
  include_missing?: boolean | undefined;
  // Hash specific
  n_features?: number | undefined;
  // Target specific
  target_column?: string | undefined;
  smooth?: number | 'auto' | undefined;
  target_type?: 'auto' | 'continuous' | 'multiclass' | 'binary' | undefined;
  // Label/Ordinal specific
  unknown_value?: number | undefined;
  missing_code?: number | undefined;
  // Ordinal specific: user-defined category order
  categories_order?: string | undefined;
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

  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics: Record<string, unknown> | null =
    nodeResult?.metrics && typeof nodeResult.metrics === 'object'
      ? (nodeResult.metrics as Record<string, unknown>)
      : null;
  const categoriesCount: Record<string, unknown> | null =
    metrics?.categories_count && typeof metrics.categories_count === 'object'
      ? (metrics.categories_count as Record<string, unknown>)
      : null;
  const classesCount: Record<string, unknown> | null =
    metrics?.classes_count && typeof metrics.classes_count === 'object'
      ? (metrics.classes_count as Record<string, unknown>)
      : null;
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
    ? Object.values(schema.columns).map(c => c.name).filter(name => !droppedUpstream.has(name))
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
          <span className="block text-sm font-medium">Encoding Method</span>
          <select
            className="w-full p-2 border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
            value={config.method}
            onChange={(e) => onChange({ ...config, method: e.target.value as EncodingConfig['method'] })}
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

          {/* Contextual behaviour hints — collapsible */}
          {/* Compact one-line behaviour badge */}
          {config.method === 'label' && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded px-2 py-1">
              <Info size={10} className="shrink-0" />
              <span>Encodes <strong>target (y)</strong> by default. Place after Feature/Target Split.</span>
            </div>
          )}
          {(config.method === 'onehot' || config.method === 'dummy' || config.method === 'hash') && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded px-2 py-1">
              <Info size={10} className="shrink-0" />
              <span><strong>Feature columns only</strong> — target is always excluded. Use Label Encoder for the target.</span>
            </div>
          )}
          {config.method === 'ordinal' && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-blue-700 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 rounded px-2 py-1">
              <Info size={10} className="shrink-0" />
              <span>Target-safe — no columns selected encodes <strong>y</strong>. Use when order matters (low/mid/high).</span>
            </div>
          )}
          {config.method === 'target' && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 rounded px-2 py-1">
              <Info size={10} className="shrink-0" />
              <span>Replaces each category with the <strong>mean of the target</strong>. Select the target column on the right.</span>
            </div>
          )}
          
          {/* drop_original only applies to OHE (the only encoder that expands to new columns) */}
          {config.method === 'onehot' && (
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
          )}
        </div>

        {/* Method Specific Settings */}
        <div className="space-y-2">
          {config.method === 'onehot' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.drop_first || false}
                  onChange={(e) => onChange({ ...config, drop_first: e.target.checked })}
                  className="rounded border-gray-300"
                />
                Drop First Category
              </label>
              <div className="space-y-1">
                <span className="block text-xs font-medium">Handle Unknown</span>
                <select
                  className="w-full p-1 text-sm border rounded"
                  value={config.handle_unknown || 'ignore'}
                  onChange={(e) => onChange({ ...config, handle_unknown: e.target.value as EncodingConfig['handle_unknown'] })}
                  title="How to handle categories seen in test data but not in training data."
                >
                  <option value="ignore">Ignore (Zeros)</option>
                  <option value="error">Raise Error</option>
                </select>
                <p className="text-[10px] text-muted-foreground">
                  &quot;Ignore&quot; produces all-zeros for unknown categories.
                </p>
              </div>
              <div className="space-y-1">
                <span className="block text-xs font-medium">Max Categories</span>
                <input
                  type="number"
                  min="2"
                  max="200"
                  className="w-full p-1 text-sm border rounded"
                  value={config.max_categories ?? 20}
                  onChange={(e) => onChange({ ...config, max_categories: parseInt(e.target.value) })}
                  title="Caps the number of one-hot columns per feature."
                />
                <p className="text-[10px] text-muted-foreground">Caps columns per feature (default 20).</p>
              </div>
              <label className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={config.include_missing || false}
                  onChange={(e) => onChange({ ...config, include_missing: e.target.checked })}
                  className="rounded border-gray-300"
                />
                Include missing as category
              </label>
            </div>
          )}

          {config.method === 'dummy' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.drop_first || false}
                  onChange={(e) => onChange({ ...config, drop_first: e.target.checked })}
                  className="rounded border-gray-300"
                />
                Drop First Category
              </label>
              <p className="text-[10px] text-muted-foreground ml-5">
                Recommended: avoids the dummy variable trap (multicollinearity).
              </p>
            </div>
          )}

          {config.method === 'target' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <span className="block text-sm font-medium">Target Column</span>
              <select
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.target_column || ''}
                onChange={(e) => onChange({ ...config, target_column: e.target.value })}
              >
                <option value="">Select a target column...</option>
                {categoricalColumns.map(name => {
                  const col = schema?.columns[name];
                  return (
                    <option key={name} value={name}>{name}{col ? ` (${col.dtype})` : ''}</option>
                  );
                })}
              </select>
              <div className="space-y-1">
                <span className="block text-xs font-medium">Smoothing</span>
                <input
                  type="text"
                  className="w-full p-1 text-sm border rounded"
                  value={config.smooth ?? 'auto'}
                  onChange={(e) => {
                    const v = e.target.value;
                    onChange({ ...config, smooth: v === 'auto' ? 'auto' : (isNaN(Number(v)) ? 'auto' : Number(v)) });
                  }}
                  placeholder="auto or number"
                  title="Smoothing strength. 'auto' uses sklearn's default."
                />
                <p className="text-[10px] text-muted-foreground">Higher = shrinks more toward global mean (default: auto).</p>
              </div>
              <div className="space-y-1">
                <span className="block text-xs font-medium">Target Type</span>
                <select
                  className="w-full p-1 text-sm border rounded"
                  value={config.target_type || 'auto'}
                  onChange={(e) => onChange({ ...config, target_type: e.target.value as EncodingConfig['target_type'] })}
                >
                  <option value="auto">Auto-detect</option>
                  <option value="continuous">Continuous (regression)</option>
                  <option value="binary">Binary classification</option>
                  <option value="multiclass">Multiclass classification</option>
                </select>
              </div>
            </div>
          )}

          {config.method === 'hash' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <span className="block text-xs font-medium">Number of Features</span>
              <input
                type="number"
                min="1"
                className="w-full p-2 border rounded"
                value={config.n_features ?? 8}
                onChange={(e) => onChange({ ...config, n_features: parseInt(e.target.value) })}
                title="Number of hash buckets."
              />
              <p className="text-[10px] text-muted-foreground">Buckets to hash categories into (default 8).</p>
            </div>
          )}

          {config.method === 'ordinal' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <div className="space-y-1">
                <span className="block text-xs font-medium">Handle Unknown</span>
                <select
                  className="w-full p-1 text-sm border rounded"
                  value={config.handle_unknown === 'ignore' || !config.handle_unknown ? 'use_encoded_value' : config.handle_unknown}
                  onChange={(e) => onChange({ ...config, handle_unknown: e.target.value as EncodingConfig['handle_unknown'] })}
                  title="How to handle categories not seen during training."
                >
                  <option value="use_encoded_value">Use encoded value (–1)</option>
                  <option value="error">Raise error</option>
                </select>
                <p className="text-[10px] text-muted-foreground">
                  &quot;Use encoded value&quot; assigns the integer below to unknown categories.
                </p>
              </div>
              <div className="space-y-1">
                <span className="block text-xs font-medium">Unknown Value</span>
                <input
                  type="number"
                  className="w-full p-2 border rounded"
                  value={config.unknown_value ?? -1}
                  disabled={config.handle_unknown === 'error'}
                  onChange={(e) => onChange({ ...config, unknown_value: parseInt(e.target.value) })}
                  title="Integer to assign for unknown categories."
                />
              </div>
              <div className="space-y-1">
                <span className="block text-xs font-medium">
                  Category Order <span className="font-normal text-muted-foreground">(optional)</span>
                </span>
                <textarea
                  className="w-full p-1.5 text-xs border rounded font-mono resize-y min-h-[56px]"
                  placeholder={"One line per column:\nlow, medium, high\ncat1, cat2, cat3"}
                  value={config.categories_order || ''}
                  rows={3}
                  onChange={(e) => onChange({ ...config, categories_order: e.target.value })}
                  title="Ordered categories per selected column. One line per column (in selection order). Leave empty for auto-detection."
                />
                <p className="text-[10px] text-muted-foreground">
                  One line per selected column, comma-separated in desired order. Leave blank for auto-detection.
                </p>
              </div>
            </div>
          )}

          {config.method === 'label' && (
            <div className="space-y-2 p-3 bg-muted/20 rounded border h-full">
              <span className="block text-sm font-medium">Missing/Unknown Code</span>
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
          <span className="block text-sm font-medium">Columns to Encode</span>
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
                No columns match &quot;{searchTerm}&quot;
              </div>
            )}
          </div>
        ) : (
          <div className="text-xs text-muted-foreground italic border rounded p-4 text-center">
            No columns found — connect an upstream dataset node.
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
                <span className="font-medium">{String(metrics.encoded_columns_count)}</span>
              </div>
            )}
            {metrics.new_features_count !== undefined && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">New Features Created:</span>
                <span className="font-medium text-primary">{String(metrics.new_features_count)}</span>
              </div>
            )}
            
            {/* Detailed Counts */}
            {categoriesCount && (
              <div className="mt-2 border-t pt-2">
                <span className="text-muted-foreground block mb-1 font-medium">Categories Found:</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {Object.entries(categoriesCount).map(([col, count]) => (
                    <div key={col} className="flex justify-between text-[10px]">
                      <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                      <span className="font-mono">{String(count)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {classesCount && (
              <div className="mt-2 border-t pt-2">
                <span className="text-muted-foreground block mb-1 font-medium">Classes Found:</span>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                  {Object.entries(classesCount).map(([col, count]) => (
                    <div key={col} className="flex justify-between text-[10px]">
                      <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                      <span className="font-mono">{String(count)}</span>
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
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const method = config.method ?? 'onehot';
    if (cols === 0) return method;
    return `${method} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: (config) => {
    // Label and Ordinal intentionally operate on y when no columns selected
    if (config.columns.length === 0 && config.method !== 'label' && config.method !== 'ordinal')
      return { isValid: false, error: 'Select at least one column' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'onehot',
    columns: [],
    drop_first: false,
    drop_original: true,
    handle_unknown: 'ignore',
    max_categories: 20,
    include_missing: false,
    n_features: 8,
    smooth: 'auto',
    target_type: 'auto',
    unknown_value: -1,
    missing_code: -1,
    categories_order: '',
  }),
};
