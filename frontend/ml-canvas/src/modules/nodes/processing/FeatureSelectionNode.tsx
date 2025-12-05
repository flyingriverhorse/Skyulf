import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Filter } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { getIncomers } from '@xyflow/react';

interface FeatureSelectionConfig {
  method: 
    | 'variance_threshold' 
    | 'correlation_threshold'
    | 'select_k_best' 
    | 'select_percentile' 
    | 'select_fpr' 
    | 'select_fdr' 
    | 'select_fwe' 
    | 'generic_univariate_select'
    | 'select_from_model' 
    | 'rfe';
  
  // Common
  target_column?: string;
  datasetId?: string;
  problem_type?: 'auto' | 'classification' | 'regression';

  // Method Specific
  threshold?: number | string; // Variance, Correlation, SelectFromModel (can be "median")
  correlation_method?: 'pearson' | 'kendall' | 'spearman';
  k?: number; // SelectKBest, RFE
  percentile?: number; // SelectPercentile
  alpha?: number; // FPR, FDR, FWE
  score_func?: string; // Univariate methods
  mode?: 'k_best' | 'percentile' | 'fpr' | 'fdr' | 'fwe'; // Generic
  param?: number; // Generic
  estimator?: 'RandomForest' | 'LogisticRegression' | 'LinearRegression' | 'auto'; // Model based
  step?: number; // RFE
  drop_columns?: boolean;
}

const FeatureSelectionSettings: React.FC<{ config: FeatureSelectionConfig; onChange: (c: FeatureSelectionConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  
  // Recursive search for datasetId
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const executionResult = useGraphStore((state) => state.executionResult);

  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
    const visited = new Set<string>();
    const queue = [currentNodeId];
    
    while (queue.length > 0) {
      const id = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      
      const node = nodes.find(n => n.id === id);
      if (!node) continue;
      
      // If this is NOT the current node, check if it has datasetId
      if (id !== currentNodeId && node.data?.datasetId) {
        return node.data.datasetId as string;
      }
      
      const incomers = getIncomers(node, nodes, edges);
      for (const incomer of incomers) {
        queue.push(incomer.id);
      }
    }
    return undefined;
  };

  const upstreamDatasetId = findUpstreamDatasetId(nodeId || '');
  const upstreamTargetColumn = upstreamData.find((d: any) => d.target_column)?.target_column as string | undefined;

  // Auto-detect target and dataset
  useEffect(() => {
    const updates: Partial<FeatureSelectionConfig> = {};
    if (upstreamDatasetId && config.datasetId !== upstreamDatasetId) {
      updates.datasetId = upstreamDatasetId;
    }
    if (upstreamTargetColumn && config.target_column !== upstreamTargetColumn) {
      updates.target_column = upstreamTargetColumn;
    }
    if (Object.keys(updates).length > 0) {
      onChange({ ...config, ...updates });
    }
  }, [upstreamDatasetId, upstreamTargetColumn, config.datasetId, config.target_column, onChange]);

  const { data: schema, isLoading } = useDatasetSchema(upstreamDatasetId || config.datasetId);
  const columns = schema ? Object.values(schema.columns).map(c => c.name) : [];

  // Responsive Layout
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
    return () => observer.disconnect();
  }, []);

  // Helper to determine available score functions based on problem type
  const getScoreFunctions = () => {
    const type = config.problem_type || 'auto';
    if (type === 'classification') {
      return [
        { value: 'f_classif', label: 'ANOVA F-value' },
        { value: 'mutual_info_classif', label: 'Mutual Information' },
        { value: 'chi2', label: 'Chi-squared' },
      ];
    } else if (type === 'regression') {
      return [
        { value: 'f_regression', label: 'F-value' },
        { value: 'mutual_info_regression', label: 'Mutual Information' },
        { value: 'r_regression', label: 'Pearson Correlation' },
      ];
    }
    return [
      { value: 'f_classif', label: 'ANOVA F-value (Classif)' },
      { value: 'f_regression', label: 'F-value (Reg)' },
      { value: 'mutual_info_classif', label: 'Mutual Info (Classif)' },
      { value: 'mutual_info_regression', label: 'Mutual Info (Reg)' },
    ];
  };

  const isUnivariate = [
    'select_k_best', 'select_percentile', 'select_fpr', 'select_fdr', 'select_fwe', 'generic_univariate_select'
  ].includes(config.method);

  const isModelBased = ['select_from_model', 'rfe'].includes(config.method);

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!upstreamDatasetId && (
          <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
            Connect a dataset node to configure.
          </div>
        )}
        {isLoading && !!upstreamDatasetId && (
          <div className="text-xs text-muted-foreground animate-pulse">Loading schema...</div>
        )}
      </div>

      {/* Main Content */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column: Method & Target */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div className="space-y-2">
            <label className="text-sm font-medium">Selection Method</label>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.method}
              onChange={(e) => onChange({ ...config, method: e.target.value as any })}
            >
              <optgroup label="Simple">
                <option value="variance_threshold">Variance Threshold</option>
                <option value="correlation_threshold">Correlation Threshold</option>
              </optgroup>
              <optgroup label="Univariate">
                <option value="select_k_best">Select K Best</option>
                <option value="select_percentile">Select Percentile</option>
                <option value="select_fpr">False Positive Rate (FPR)</option>
                <option value="select_fdr">False Discovery Rate (FDR)</option>
                <option value="select_fwe">Family-wise Error (FWE)</option>
                <option value="generic_univariate_select">Generic Univariate</option>
              </optgroup>
              <optgroup label="Model Based">
                <option value="select_from_model">Select From Model</option>
                <option value="rfe">Recursive Feature Elimination (RFE)</option>
              </optgroup>
            </select>
          </div>

          {config.method !== 'variance_threshold' && config.method !== 'correlation_threshold' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Target Column</label>
              {upstreamTargetColumn ? (
                <div className="p-2 bg-muted rounded text-sm text-muted-foreground border">
                  {upstreamTargetColumn} <span className="text-xs italic">(Auto-detected)</span>
                </div>
              ) : (
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.target_column ?? ''}
                  onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                >
                  <option value="">Select Target...</option>
                  {columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              )}
              <p className="text-xs text-muted-foreground">Required for supervised selection methods.</p>
            </div>
          )}

          {(isUnivariate || isModelBased) && (
             <div className="space-y-2">
               <label className="text-sm font-medium">Problem Type</label>
               <select
                 className="w-full p-2 border rounded bg-background text-sm"
                 value={config.problem_type ?? 'auto'}
                 onChange={(e) => onChange({ ...config, problem_type: e.target.value as any })}
               >
                 <option value="auto">Auto-detect</option>
                 <option value="classification">Classification</option>
                 <option value="regression">Regression</option>
               </select>
             </div>
          )}
        </div>

        {/* Right Column: Parameters */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l' : 'shrink-0 border-t pt-4'}`}>
          <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Parameters</h4>
          
          {/* Variance Threshold */}
          {config.method === 'variance_threshold' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Threshold</label>
              <input
                type="number"
                step="0.01"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.threshold ?? 0}
                onChange={(e) => onChange({ ...config, threshold: parseFloat(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Features with variance lower than this will be removed.</p>
            </div>
          )}

          {/* Correlation Threshold */}
          {config.method === 'correlation_threshold' && (
            <>
              <div className="space-y-2">
                <label className="text-sm font-medium">Threshold</label>
                <input
                  type="number"
                  step="0.01"
                  max="1"
                  min="0"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.threshold ?? 0.95}
                  onChange={(e) => onChange({ ...config, threshold: parseFloat(e.target.value) })}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Method</label>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.correlation_method ?? 'pearson'}
                  onChange={(e) => onChange({ ...config, correlation_method: e.target.value as any })}
                >
                  <option value="pearson">Pearson</option>
                  <option value="spearman">Spearman</option>
                  <option value="kendall">Kendall</option>
                </select>
              </div>
            </>
          )}

          {/* Univariate Common: Score Function */}
          {isUnivariate && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Scoring Function</label>
              <select
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.score_func ?? ''}
                onChange={(e) => onChange({ ...config, score_func: e.target.value })}
              >
                <option value="">Auto (Default)</option>
                {getScoreFunctions().map(f => (
                  <option key={f.value} value={f.value}>{f.label}</option>
                ))}
              </select>
            </div>
          )}

          {/* K Best / RFE */}
          {(config.method === 'select_k_best' || config.method === 'rfe') && (
            <div className="space-y-2">
              <label className="text-sm font-medium">K (Number of Features)</label>
              <input
                type="number"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.k ?? 10}
                onChange={(e) => onChange({ ...config, k: parseInt(e.target.value) })}
              />
            </div>
          )}

          {/* Percentile */}
          {config.method === 'select_percentile' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Percentile</label>
              <input
                type="number"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.percentile ?? 10}
                onChange={(e) => onChange({ ...config, percentile: parseInt(e.target.value) })}
              />
            </div>
          )}

          {/* Alpha (FPR, FDR, FWE) */}
          {['select_fpr', 'select_fdr', 'select_fwe'].includes(config.method) && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Alpha (Significance)</label>
              <input
                type="number"
                step="0.001"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.alpha ?? 0.05}
                onChange={(e) => onChange({ ...config, alpha: parseFloat(e.target.value) })}
              />
            </div>
          )}

          {/* Generic Univariate */}
          {config.method === 'generic_univariate_select' && (
            <>
              <div className="space-y-2">
                <label className="text-sm font-medium">Mode</label>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.mode ?? 'k_best'}
                  onChange={(e) => onChange({ ...config, mode: e.target.value as any })}
                >
                  <option value="k_best">K Best</option>
                  <option value="percentile">Percentile</option>
                  <option value="fpr">FPR</option>
                  <option value="fdr">FDR</option>
                  <option value="fwe">FWE</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Parameter</label>
                <input
                  type="number"
                  step="0.001"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.param ?? 1e-5}
                  onChange={(e) => onChange({ ...config, param: parseFloat(e.target.value) })}
                />
                <p className="text-xs text-muted-foreground">Value for the selected mode (e.g., k, percentile, or alpha).</p>
              </div>
            </>
          )}

          {/* Model Based Common */}
          {isModelBased && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Estimator</label>
              <select
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.estimator ?? 'auto'}
                onChange={(e) => onChange({ ...config, estimator: e.target.value as any })}
              >
                <option value="auto">Auto</option>
                <option value="RandomForest">Random Forest</option>
                <option value="LogisticRegression">Logistic Regression</option>
                <option value="LinearRegression">Linear Regression</option>
              </select>
            </div>
          )}

          {/* Select From Model Specific */}
          {config.method === 'select_from_model' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Threshold</label>
              <input
                type="text"
                className="w-full p-2 border rounded bg-background text-sm"
                placeholder="e.g., median, mean, 1.25*mean"
                value={config.threshold ?? 'median'}
                onChange={(e) => onChange({ ...config, threshold: e.target.value })}
              />
              <p className="text-xs text-muted-foreground">String (e.g. "median") or float.</p>
            </div>
          )}

          {/* RFE Specific */}
          {config.method === 'rfe' && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Step</label>
              <input
                type="number"
                min="1"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.step ?? 1}
                onChange={(e) => onChange({ ...config, step: parseInt(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Features to remove at each iteration.</p>
            </div>
          )}

          {/* Drop Columns Checkbox */}
          <div className="flex items-center space-x-2 pt-2 border-t">
            <input
              type="checkbox"
              id="drop_columns"
              className="rounded border-gray-300"
              checked={config.drop_columns !== false}
              onChange={(e) => onChange({ ...config, drop_columns: e.target.checked })}
            />
            <label htmlFor="drop_columns" className="text-sm font-medium">
              Drop Columns
            </label>
          </div>
          <p className="text-xs text-muted-foreground">
            If unchecked, columns will be identified but not removed from the dataset.
          </p>

          {/* Feedback Section */}
          {executionResult && executionResult.node_results[nodeId || ''] && (
            <div className="mt-4 p-3 bg-muted/50 rounded border text-xs">
              {(() => {
                const result = executionResult.node_results[nodeId || ''];
                const dropped = result.metrics?.dropped_columns as string[] | undefined;

                return (
                  <div className="space-y-2">
                    {result.error && (
                      <div className="p-2 bg-destructive/10 text-destructive rounded border border-destructive/20">
                        {result.error}
                      </div>
                    )}

                    {dropped && dropped.length > 0 && (
                      <div>
                        <div className="font-medium text-muted-foreground mb-1">Dropped Columns ({dropped.length}):</div>
                        <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border">
                          <ul className="list-disc list-inside space-y-0.5 text-muted-foreground">
                            {dropped.map(col => (
                              <li key={col} className="truncate" title={col}>{col}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    )}

                    {dropped && dropped.length === 0 && result.status === 'success' && (
                      <div className="text-muted-foreground italic">No columns were dropped.</div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const FeatureSelectionNode: NodeDefinition<FeatureSelectionConfig> = {
  type: 'feature_selection',
  label: 'Feature Selection',
  category: 'Preprocessing',
  description: 'Select the most important features.',
  icon: Filter,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Selected', type: 'dataset' }],
  settings: FeatureSelectionSettings,
  validate: (config) => {
    if (config.method !== 'variance_threshold' && !config.target_column) {
      return { isValid: false, message: 'Target column is required for this method.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'select_k_best',
    k: 10,
  }),
};
