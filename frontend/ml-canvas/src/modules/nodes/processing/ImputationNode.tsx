import React, { useState, useMemo, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { PaintBucket } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { Recommendation } from '../../../core/api/client';
import { useGraphStore } from '../../../core/store/useGraphStore';

interface ImputationConfig {
  columns: string[];
  method: 'simple' | 'knn' | 'iterative';
  
  // Simple Imputer
  strategy: 'mean' | 'median' | 'most_frequent' | 'constant';
  fill_value?: string | number;

  // KNN Imputer
  n_neighbors?: number;
  weights?: 'uniform' | 'distance';

  // Iterative Imputer
  max_iter?: number;
  estimator?: 'bayesian_ridge' | 'decision_tree' | 'extra_trees' | 'knn';
  random_state?: number;
}

const ImputationSettings: React.FC<{ config: ImputationConfig; onChange: (c: ImputationConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns).map(c => c.name) : [];
  
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
  
  const recommendations = useRecommendations(nodeId || '', {
    types: ['imputation'],
    suggestedNodeTypes: ['SimpleImputer', 'KNNImputer', 'IterativeImputer'],
    scope: 'column'
  });

  const handleApplyRecommendation = (rec: Recommendation) => {
    if (rec.target_columns && rec.target_columns.length > 0) {
      const newCols = Array.from(new Set([...config.columns, ...rec.target_columns]));
      // Apply strategy if recommended (assuming recommendation might contain params)
      // For now just columns
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

  const FeedbackSection = () => (
    metrics ? (
      <div className="mt-4 p-3 bg-muted/50 rounded border text-xs">
        <div className="font-medium text-muted-foreground mb-2">Execution Feedback</div>
        
        {metrics.fill_values && (
          <div className="space-y-1">
            <span className="text-muted-foreground block mb-1 font-medium">Imputed Values:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
              {Object.entries(metrics.fill_values).map(([col, val]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {metrics.missing_counts && (
          <div className="space-y-1 mt-2">
            <span className="text-muted-foreground block mb-1 font-medium">Missing Values Filled:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
              {Object.entries(metrics.missing_counts).map(([col, count]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{String(count)}</span>
                </div>
              ))}
            </div>
            {metrics.total_missing !== undefined && (
               <div className="text-[10px] text-muted-foreground mt-1 pt-1 border-t">
                  Total Filled: <span className="font-mono font-medium">{metrics.total_missing}</span>
               </div>
            )}
          </div>
        )}
        
        {/* Generic success message if no specific metrics but successful */}
        {!metrics.fill_values && !metrics.missing_counts && nodeResult?.status === 'success' && (
          <div className="text-green-600">Imputation completed successfully.</div>
        )}
      </div>
    ) : null
  );

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
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

      {/* Main Content Area */}
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

          <div>
            <label className="block text-sm font-medium mb-1">Imputation Method</label>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.method || 'simple'}
              onChange={(e) => onChange({ ...config, method: e.target.value as any })}
            >
              <option value="simple">Simple Imputer (Univariate)</option>
              <option value="knn">KNN Imputer (Multivariate)</option>
              <option value="iterative">Iterative Imputer (MICE)</option>
            </select>
          </div>

          {/* Simple Imputer Settings */}
          {(config.method === 'simple' || !config.method) && (
            <>
              <div>
                <label className="block text-sm font-medium mb-1">Strategy</label>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.strategy}
                  onChange={(e) => onChange({ ...config, strategy: e.target.value as any })}
                >
                  <option value="mean">Mean (Average)</option>
                  <option value="median">Median (Middle Value)</option>
                  <option value="most_frequent">Most Frequent (Mode)</option>
                  <option value="constant">Constant Value</option>
                </select>
                <p className="text-[10px] text-muted-foreground mt-1">
                  {config.strategy === 'mean' && 'Replaces missing values with the mean of the column. (Numeric only)'}
                  {config.strategy === 'median' && 'Replaces missing values with the median. (Robust to outliers)'}
                  {config.strategy === 'most_frequent' && 'Replaces missing with the most common value. (Categorical/Numeric)'}
                  {config.strategy === 'constant' && 'Replaces missing values with a specific value.'}
                </p>
              </div>

              {config.strategy === 'constant' && (
                <div>
                  <label className="block text-sm font-medium mb-1">Fill Value</label>
                  <input
                    type="text"
                    className="w-full p-2 border rounded bg-background text-sm"
                    value={config.fill_value || ''}
                    onChange={(e) => onChange({ ...config, fill_value: e.target.value })}
                    placeholder="Enter value..."
                  />
                </div>
              )}
            </>
          )}

          {/* KNN Imputer Settings */}
          {config.method === 'knn' && (
            <>
              <div>
                <label className="block text-sm font-medium mb-1">Number of Neighbors</label>
                <input
                  type="number"
                  min="1"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.n_neighbors || 5}
                  onChange={(e) => onChange({ ...config, n_neighbors: parseInt(e.target.value) })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Weights</label>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.weights || 'uniform'}
                  onChange={(e) => onChange({ ...config, weights: e.target.value as any })}
                >
                  <option value="uniform">Uniform</option>
                  <option value="distance">Distance</option>
                </select>
                <p className="text-[10px] text-muted-foreground mt-1">
                  Uniform: All points in each neighborhood are weighted equally.
                  Distance: Weight points by the inverse of their distance.
                </p>
              </div>
            </>
          )}

          {/* Iterative Imputer Settings */}
          {config.method === 'iterative' && (
            <>
              <div>
                <label className="block text-sm font-medium mb-1">Max Iterations</label>
                <input
                  type="number"
                  min="1"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.max_iter || 10}
                  onChange={(e) => onChange({ ...config, max_iter: parseInt(e.target.value) })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Estimator</label>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.estimator || 'bayesian_ridge'}
                  onChange={(e) => onChange({ ...config, estimator: e.target.value as any })}
                >
                  <option value="bayesian_ridge">Bayesian Ridge</option>
                  <option value="decision_tree">Decision Tree</option>
                  <option value="extra_trees">Extra Trees</option>
                  <option value="knn">KNN</option>
                </select>
                <p className="text-[10px] text-muted-foreground mt-1">
                  The estimator to use for the imputation steps.
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Random State</label>
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.random_state ?? 0}
                  onChange={(e) => onChange({ ...config, random_state: parseInt(e.target.value) })}
                />
              </div>
            </>
          )}

          {/* Feedback Section - Only show here if wide */}
          {isWide && <FeedbackSection />}
        </div>

        {/* Right Column (Column Selection) */}
        <div className={`flex flex-col h-full min-h-[200px] border rounded-md overflow-hidden ${isWide ? '' : 'shrink-0'}`}>
          <div className="p-2 border-b bg-muted/30 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">Target Columns ({config.columns.length})</span>
              <div className="flex gap-1">
                <button onClick={handleSelectAll} className="text-[10px] px-2 py-1 hover:bg-accent rounded">All</button>
                <button onClick={handleDeselectAll} className="text-[10px] px-2 py-1 hover:bg-accent rounded">None</button>
              </div>
            </div>
            <input
              type="text"
              placeholder="Search columns..."
              className="w-full text-xs p-1.5 border rounded bg-background"
              value={searchTerm}
              onChange={(e) => { setSearchTerm(e.target.value); }}
            />
          </div>
          
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredColumns.length > 0 ? (
              filteredColumns.map(col => (
                <label key={col} className="flex items-center justify-between gap-2 text-sm hover:bg-accent/50 p-1.5 rounded cursor-pointer select-none">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <input
                      type="checkbox"
                      checked={config.columns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...config.columns, col]
                          : config.columns.filter(c => c !== col);
                        onChange({ ...config, columns: newCols });
                      }}
                      className="rounded border-gray-300 text-primary focus:ring-primary shrink-0"
                    />
                    <span className="truncate" title={col}>{col}</span>
                  </div>
                  {metrics?.missing_counts && metrics.missing_counts[col] !== undefined && (
                    <span className="text-[10px] text-muted-foreground font-mono shrink-0 bg-muted px-1.5 py-0.5 rounded" title={`${metrics.missing_counts[col]} missing values filled`}>
                      {metrics.missing_counts[col]}
                    </span>
                  )}
                </label>
              ))
            ) : (
              <div className="p-4 text-center text-xs text-muted-foreground">
                {availableColumns.length === 0 ? 'No columns available' : 'No matches found'}
              </div>
            )}
          </div>
        </div>
        
        {/* Feedback Section - Show here if NOT wide (mobile/narrow) */}
        {!isWide && <FeedbackSection />}
      </div>
    </div>
  );
};

export const ImputationNode: NodeDefinition<ImputationConfig> = {
  type: 'imputation_node',
  label: 'Imputation',
  category: 'Preprocessing',
  description: 'Fill missing values using Simple, KNN, or Iterative strategies.',
  icon: PaintBucket,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: ImputationSettings,
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column' };
    if ((config.method === 'simple' || !config.method) && config.strategy === 'constant' && (config.fill_value === undefined || config.fill_value === '')) {
      return { isValid: false, message: 'Fill value is required for Constant strategy' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    method: 'simple',
    strategy: 'mean',
    fill_value: 0,
    n_neighbors: 5,
    weights: 'uniform',
    max_iter: 10,
    estimator: 'bayesian_ridge',
    random_state: 0
  })
};
