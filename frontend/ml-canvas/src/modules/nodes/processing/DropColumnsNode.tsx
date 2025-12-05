import React, { useState, useMemo, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Trash2 } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { Recommendation } from '../../../core/api/client';

interface DropColumnsConfig {
  columns: string[];
  missing_threshold?: number; // 0-100
}

const DropColumnsSettings: React.FC<{ config: DropColumnsConfig; onChange: (c: DropColumnsConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns).map(c => c.name) : [];
  
  // Responsive Layout Logic
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > 450); // Switch to 2-column layout if wider than 450px
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);
  
  const recommendations = useRecommendations(nodeId || '', {
    types: ['cleaning', 'feature_selection'],
    suggestedNodeTypes: ['DropMissingColumns'],
    scope: 'column'
  });

  const handleApplyRecommendation = (rec: Recommendation) => {
    if (rec.target_columns && rec.target_columns.length > 0) {
      const newCols = Array.from(new Set([...config.columns, ...rec.target_columns]));
      onChange({ ...config, columns: newCols });
    }
  };
  
  const [searchTerm, setSearchTerm] = useState('');

  const filteredColumns = useMemo(() => {
    return availableColumns.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [availableColumns, searchTerm]);

  const handleSelectAll = () => {
    onChange({ ...config, columns: filteredColumns });
  };

  const handleDeselectAll = () => {
    const newCols = config.columns.filter(c => !filteredColumns.includes(c));
    onChange({ ...config, columns: newCols });
  };

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar (Always Visible) */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
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
      </div>

      {/* Main Content Area - Responsive Grid/Flex */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column (Settings) */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          {/* Recommendations */}
          {recommendations.length > 0 && (
            <div className="shrink-0">
              <RecommendationsPanel 
                recommendations={recommendations} 
                onApply={handleApplyRecommendation}
              />
            </div>
          )}

          {/* Threshold Section */}
          <div className="space-y-3 p-3 border rounded-md bg-muted/5">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <label className="text-sm font-medium text-foreground">Missing Value Threshold</label>
                <p className="text-[10px] text-muted-foreground">
                  Drop columns with &gt;{config.missing_threshold ?? 0}% missing values
                </p>
              </div>
              <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">
                {config.missing_threshold ?? 0}%
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={config.missing_threshold ?? 0}
              onChange={(e) => onChange({ ...config, missing_threshold: parseInt(e.target.value) })}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
            />
          </div>
        </div>

        {/* Right Column (Column List) */}
        <div className={`flex flex-col border rounded-md bg-background shadow-sm overflow-hidden ${isWide ? 'min-h-0 flex-1' : 'h-96 shrink-0'}`}>
          <div className="p-3 border-b bg-muted/5 space-y-3 shrink-0">
            <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Explicitly Drop Columns</label>
                <span className="text-xs text-muted-foreground">
                  {config.columns.length} selected
                </span>
            </div>
            
            {/* Search */}
            <div className="relative group">
              <input 
                type="text" 
                placeholder="Search columns..." 
                className="block w-full pl-9 pr-3 py-1.5 text-sm bg-background border rounded-md shadow-sm focus:ring-1 focus:ring-primary focus:border-primary transition-all outline-none"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="flex gap-2 text-xs justify-end">
                <button onClick={handleSelectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Select All</button>
                <span className="text-border">|</span>
                <button onClick={handleDeselectAll} className="text-primary hover:text-primary/80 font-medium transition-colors">Deselect All</button>
            </div>
          </div>

          {/* Scrollable List */}
          <div className="flex-1 overflow-y-auto p-1">
            {availableColumns.length > 0 ? (
              filteredColumns.length > 0 ? (
                <div className="space-y-0.5">
                  {filteredColumns.map(col => (
                    <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-2 rounded cursor-pointer transition-colors select-none">
                      <input
                        type="checkbox"
                        checked={config.columns.includes(col)}
                        onChange={(e) => {
                          const newCols = e.target.checked
                            ? [...config.columns, col]
                            : config.columns.filter(c => c !== col);
                          onChange({ ...config, columns: newCols });
                        }}
                        className="rounded border-gray-300 text-primary focus:ring-primary w-4 h-4"
                      />
                      <span className="truncate font-mono text-xs" title={col}>{col}</span>
                    </label>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground text-center py-8">
                  No columns match "{searchTerm}"
                </div>
              )
            ) : (
              <div className="text-xs text-muted-foreground italic p-8 text-center">
                No columns available
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export const DropColumnsNode: NodeDefinition<DropColumnsConfig> = {
  type: 'drop_missing_columns', // Maps to backend handler
  label: 'Drop Columns',
  category: 'Preprocessing',
  description: 'Remove specific columns or those with high missing values.',
  icon: Trash2,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Data', type: 'dataset' }],
  settings: DropColumnsSettings,
  validate: (config) => ({ 
    isValid: config.columns.length > 0 || (config.missing_threshold !== undefined && config.missing_threshold > 0),
    message: (config.columns.length === 0 && (!config.missing_threshold || config.missing_threshold === 0)) 
      ? 'Select columns or set a threshold' 
      : undefined
  }),
  getDefaultConfig: () => ({
    columns: [],
    missing_threshold: 0
  }),
};
