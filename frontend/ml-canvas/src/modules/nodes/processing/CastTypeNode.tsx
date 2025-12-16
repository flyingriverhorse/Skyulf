import React, { useRef, useState, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { FileType2, Plus, Trash2, Activity } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface CastTypeConfig {
  column_types: Record<string, string>;
}

const TARGET_TYPES = [
  { value: 'float', label: 'Float (Decimal)' },
  { value: 'int', label: 'Integer' },
  { value: 'string', label: 'String (Text)' },
  { value: 'category', label: 'Category' },
  { value: 'bool', label: 'Boolean' },
  { value: 'datetime', label: 'Datetime' },
];

const CastTypeSettings: React.FC<{ config: CastTypeConfig; onChange: (c: CastTypeConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics = nodeResult?.metrics;

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

  const handleAdd = () => {
    if (columns.length === 0) return;
    // Default to first available column and float
    const newCol = columns.find(c => !config.column_types[c]) || columns[0];
    onChange({
      ...config,
      column_types: { ...config.column_types, [newCol]: 'float' }
    });
  };

  const handleRemove = (col: string) => {
    const newTypes = { ...config.column_types };
    delete newTypes[col];
    onChange({ ...config, column_types: newTypes });
  };

  const handleUpdateColumn = (oldCol: string, newCol: string) => {
    if (oldCol === newCol) return;
    const type = config.column_types[oldCol];
    const newTypes = { ...config.column_types };
    delete newTypes[oldCol];
    newTypes[newCol] = type;
    onChange({ ...config, column_types: newTypes });
  };

  const handleUpdateType = (col: string, newType: string) => {
    onChange({
      ...config,
      column_types: { ...config.column_types, [col]: newType }
    });
  };

  return (
    <div ref={containerRef} className="p-4 space-y-4 h-full overflow-y-auto">
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

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Casting Rules</label>
          <button
            onClick={handleAdd}
            disabled={!datasetId || columns.length === 0}
            className="p-1.5 bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 transition-colors"
            title="Add Casting Rule"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        <div className={`space-y-2 ${isWide ? 'grid grid-cols-2 gap-2 space-y-0' : ''}`}>
          {Object.entries(config.column_types).map(([col, type]) => (
            <div key={col} className="flex items-center gap-2 p-2 border rounded bg-muted/10 group">
              <div className="flex-1 space-y-2 min-w-0">
                <select
                  className="w-full p-1.5 text-sm border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
                  value={col}
                  onChange={(e) => { handleUpdateColumn(col, e.target.value); }}
                >
                  {columns.map(c => (
                    <option key={c} value={c} disabled={c !== col && !!config.column_types[c]}>
                      {c}
                    </option>
                  ))}
                </select>
                <select
                  className="w-full p-1.5 text-sm border rounded bg-background focus:ring-1 focus:ring-primary outline-none"
                  value={type}
                  onChange={(e) => { handleUpdateType(col, e.target.value); }}
                >
                  {TARGET_TYPES.map(t => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={() => { handleRemove(col); }}
                className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded transition-colors"
                title="Remove Rule"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
          
          {Object.keys(config.column_types).length === 0 && (
            <div className={`text-xs text-muted-foreground italic border border-dashed rounded p-8 text-center ${isWide ? 'col-span-2' : ''}`}>
              No casting rules defined. Click + to add one.
            </div>
          )}
        </div>
      </div>

      {/* Feedback Section */}
      {metrics && (metrics.cast_errors !== undefined || metrics.casted_columns_count !== undefined) && (
        <div className="mt-4 p-3 bg-muted/30 rounded-md border border-border">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-primary">
            <Activity size={14} />
            <span>Last Run Results</span>
          </div>
          <div className="space-y-1 text-xs">
            {metrics.casted_columns_count !== undefined && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Columns Casted:</span>
                <span className="font-medium">{metrics.casted_columns_count}</span>
              </div>
            )}
            {metrics.cast_errors !== undefined && (
              <div className="flex justify-between">
                <span className="text-muted-foreground">Errors Encountered:</span>
                <span className="font-medium text-destructive">{metrics.cast_errors}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export const CastTypeNode: NodeDefinition<CastTypeConfig> = {
  type: 'casting',
  label: 'Cast Types',
  category: 'Preprocessing',
  description: 'Cast columns to specific data types.',
  icon: FileType2,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Casted Data', type: 'dataset' }],
  settings: CastTypeSettings,
  validate: (_config) => ({ isValid: true }),
  getDefaultConfig: () => ({
    column_types: {},
  }),
};
