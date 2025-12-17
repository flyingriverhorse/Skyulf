import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Scissors } from 'lucide-react';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { getIncomers } from '@xyflow/react';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { Recommendation } from '../../../core/api/client';

interface OutlierConfig {
  method: 'iqr' | 'zscore' | 'winsorize' | 'elliptic_envelope';
  columns: string[];
  
  // IQR
  multiplier?: number;
  
  // Z-Score
  threshold?: number;
  
  // Winsorize
  lower_percentile?: number;
  upper_percentile?: number;
  
  // Elliptic Envelope
  contamination?: number;
}

const OutlierSettings: React.FC<{ config: OutlierConfig; onChange: (c: OutlierConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  // Recursive search for datasetId
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  
  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
    const visited = new Set<string>();
    const queue = [currentNodeId];
    
    while (queue.length > 0) {
      const id = queue.shift();
      if (!id) continue;
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

  const datasetId = findUpstreamDatasetId(nodeId || '');
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  
  const backendRecommendations = useRecommendations(nodeId || '', {
    types: ['outlier_removal', 'cleaning'],
    suggestedNodeTypes: ['OutlierRemoval', 'outlier'],
    scope: 'column'
  });

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  
  // Responsive Layout
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

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

  // Filter for numeric columns only
  const numericColumns = schema 
    ? Object.values(schema.columns)
        .filter(c => ['int', 'float', 'number'].some(t => c.dtype.toLowerCase().includes(t)))
        .map(c => c.name)
    : [];

  const filteredColumns = numericColumns.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));

  const handleSelectAll = () => {
    onChange({ ...config, columns: filteredColumns });
  };

  const handleDeselectAll = () => {
    const newCols = config.columns.filter(c => !filteredColumns.includes(c));
    onChange({ ...config, columns: newCols });
  };

  const FeedbackSection = () => {
    if (!nodeResult || !nodeResult.metrics) return null;
    const m = nodeResult.metrics;
    
    // Check for rows removed (common for IQR, ZScore, Elliptic)
    const rowsRemoved = m.rows_removed as number | undefined;
    const rowsRemaining = m.rows_remaining as number | undefined;
    const rowsTotal = m.rows_total as number | undefined ?? ((rowsRemoved ?? 0) + (rowsRemaining ?? 0));
    
    // Check for bounds (IQR, Winsorize)
    const bounds = m.bounds as Record<string, { lower: number, upper: number }> | undefined;
    
    // Check for stats (ZScore)
    const stats = m.stats as Record<string, { mean: number, std: number }> | undefined;

    // Check for contamination (Elliptic Envelope)
    const contamination = m.contamination as number | undefined;

    // Check for warnings
    const warnings = m.warnings as string[] | undefined;

    // Check for values clipped (Winsorize)
    const valuesClipped = m.values_clipped as number | undefined;

    if (rowsRemoved === undefined && !bounds && !stats && contamination === undefined && !warnings && valuesClipped === undefined) return null;

    const getTargetExplanation = () => {
        switch (config.method) {
            case 'iqr': return "Removes statistical outliers (IQR)";
            case 'zscore': return "Removes deviations > 3σ";
            case 'winsorize': return "Clips extreme values";
            case 'elliptic_envelope': return "Detects multivariate anomalies";
            default: return "";
        }
    };

    return (
      <div className="mt-4 p-3 bg-muted/50 rounded border text-xs space-y-3">
        <div className="flex justify-between items-center">
            <div className="font-medium text-muted-foreground">Execution Feedback</div>
            <div className="text-[10px] text-primary/80 bg-primary/5 px-1.5 py-0.5 rounded border border-primary/10" title="Goal of the selected method">
                {getTargetExplanation()}
            </div>
        </div>
        
        {warnings && warnings.length > 0 && (
          <div className="p-2 bg-yellow-50 text-yellow-800 rounded border border-yellow-200">
            <div className="font-medium mb-1">Warnings</div>
            <ul className="list-disc list-inside space-y-0.5">
              {warnings.map((w, i) => (
                <li key={i} className="truncate" title={w}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        
        {config.method === 'winsorize' ? (
             <div className="flex justify-between items-center p-2 bg-background rounded border">
                <span className="text-muted-foreground">Values Clipped</span>
                <div className="text-right">
                    <span className="font-mono font-medium text-blue-600">{valuesClipped ?? 0}</span>
                    <span className="text-[10px] text-muted-foreground block">
                        (Replaced with bounds)
                    </span>
                </div>
             </div>
        ) : (
            (rowsRemoved !== undefined || rowsTotal !== undefined) && (
            <div className="flex justify-between items-center p-2 bg-background rounded border">
                <span className="text-muted-foreground">Rows Removed</span>
                <div className="text-right">
                <div className="flex items-center justify-end gap-1">
                    <span className="font-mono font-medium text-destructive">{rowsRemoved ?? 0}</span>
                    {rowsTotal > 0 && (
                        <span className="text-[10px] text-muted-foreground">
                            ({(((rowsRemoved ?? 0) / rowsTotal) * 100).toFixed(1)}%)
                        </span>
                    )}
                </div>
                {rowsRemaining !== undefined && (
                    <span className="text-[10px] text-muted-foreground block">
                        {rowsRemaining} remaining
                    </span>
                )}
                </div>
            </div>
            )
        )}

        {contamination !== undefined && (
           <div className="flex justify-between items-center p-2 bg-background rounded border">
            <span className="text-muted-foreground">Contamination</span>
            <span className="font-mono font-medium">{(contamination * 100).toFixed(1)}%</span>
          </div>
        )}

        {bounds && (
          <div>
            <div className="font-medium text-muted-foreground mb-1">Calculated Bounds</div>
            <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border space-y-1">
              {Object.entries(bounds).map(([col, bound]) => (
                <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
                  <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    [{bound.lower.toFixed(2)}, {bound.upper.toFixed(2)}]
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {stats && (
          <div>
            <div className="font-medium text-muted-foreground mb-1">Z-Score Stats</div>
            <div className="max-h-32 overflow-y-auto bg-background p-2 rounded border space-y-1">
              {Object.entries(stats).map(([col, stat]) => (
                <div key={col} className="flex justify-between items-center border-b border-border/50 last:border-0 pb-1 last:pb-0">
                  <span className="truncate max-w-[100px] font-medium" title={col}>{col}</span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    μ={stat.mean.toFixed(2)}, σ={stat.std.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const RecommendationsSection = () => {
    // Combine backend recommendations with runtime feedback
    const runtimeRecommendations: Recommendation[] = [];
    
    if (nodeResult && nodeResult.metrics) {
        const m = nodeResult.metrics;
        const rowsRemoved = m.rows_removed as number | undefined;
        const rowsTotal = m.rows_total as number | undefined;
        
        // If rowsTotal is missing but we have rowsRemoved, we can still give some feedback
        const effectiveRowsTotal = rowsTotal ?? (m[`${config.method === 'elliptic_envelope' ? 'EllipticEnvelope' : config.method === 'zscore' ? 'ZScore' : config.method === 'winsorize' ? 'Winsorize' : 'IQR'}_rows_total`] as number | undefined);

        if (rowsRemoved !== undefined && effectiveRowsTotal !== undefined && effectiveRowsTotal > 0) {
            const lossRatio = rowsRemoved / effectiveRowsTotal;
            if (lossRatio > 0.2) {
                runtimeRecommendations.push({
                    rule_id: 'high_data_loss',
                    type: 'warning',
                    description: "High data loss (>20%).",
                    reasoning: "Consider relaxing parameters (e.g., higher IQR multiplier or Z-Score threshold) or using Winsorization to clip values instead of removing rows.",
                    suggested_node_type: 'OutlierRemoval',
                    suggested_params: {},
                    confidence: 1.0,
                    target_columns: []
                });
            } else if (lossRatio === 0) {
                if (config.method !== 'winsorize') {
                    runtimeRecommendations.push({
                        rule_id: 'no_outliers',
                        type: 'info',
                        description: "No outliers detected.",
                        reasoning: "If you suspect outliers, try tightening the parameters (e.g., lower IQR multiplier).",
                        suggested_node_type: 'OutlierRemoval',
                        suggested_params: {},
                        confidence: 1.0,
                        target_columns: []
                    });
                }
            }
        }

        if (config.method === 'elliptic_envelope') {
            runtimeRecommendations.push({
                rule_id: 'elliptic_stochastic',
                type: 'info',
                description: "Stochastic Method",
                reasoning: "Elliptic Envelope is stochastic. Results may vary slightly between runs.",
                suggested_node_type: 'OutlierRemoval',
                suggested_params: {},
                confidence: 1.0,
                target_columns: []
            });
        }
    }

    const allRecommendations = [...backendRecommendations, ...runtimeRecommendations];

    if (allRecommendations.length === 0) return null;

    return (
        <div className="mt-4">
            <RecommendationsPanel recommendations={allRecommendations} />
        </div>
    );
  };

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
      <div className="shrink-0 p-4 pb-0 space-y-2">
        {!datasetId && (
          <div className="p-2 bg-yellow-50 text-yellow-800 text-xs rounded border border-yellow-200">
            Connect a dataset node to see available columns.
          </div>
        )}
      </div>
      
      {/* Main Content */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column: Settings */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <div>
            <label className="block text-sm font-medium mb-1">Method</label>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.method}
              onChange={(e) => onChange({ ...config, method: e.target.value as OutlierConfig['method'] })}
            >
              <option value="iqr">IQR (Interquartile Range)</option>
              <option value="zscore">Z-Score (Standard Deviation)</option>
              <option value="winsorize">Winsorize (Clip Values)</option>
              <option value="elliptic_envelope">Elliptic Envelope (Multivariate)</option>
            </select>
            <p className="text-[10px] text-muted-foreground mt-1">
              {config.method === 'iqr' && 'Removes rows with values outside Q1/Q3 ± multiplier * IQR.'}
              {config.method === 'zscore' && 'Removes rows with values more than N standard deviations from mean.'}
              {config.method === 'winsorize' && 'Clips values to specified percentiles instead of removing rows.'}
              {config.method === 'elliptic_envelope' && 'Fits a robust covariance estimate to detect outliers.'}
            </p>
          </div>

          {/* IQR Params */}
          {config.method === 'iqr' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Multiplier</label>
              <input
                type="number"
                step="0.1"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.multiplier ?? 1.5}
                onChange={(e) => onChange({ ...config, multiplier: parseFloat(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Usually 1.5 for outliers, 3.0 for extreme outliers.</p>
            </div>
          )}

          {/* Z-Score Params */}
          {config.method === 'zscore' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Threshold (Sigma)</label>
              <input
                type="number"
                step="0.1"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.threshold ?? 3.0}
                onChange={(e) => onChange({ ...config, threshold: parseFloat(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Number of standard deviations to tolerate.</p>
            </div>
          )}

          {/* Winsorize Params */}
          {config.method === 'winsorize' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Percentiles</label>
              <div className="flex gap-2 items-center">
                <div className="flex-1">
                  <label className="text-[10px] text-muted-foreground">Lower</label>
                  <input
                    type="number"
                    className="w-full p-2 border rounded bg-background text-sm"
                    value={config.lower_percentile ?? 5.0}
                    onChange={(e) => onChange({ ...config, lower_percentile: parseFloat(e.target.value) })}
                  />
                </div>
                <div className="flex-1">
                  <label className="text-[10px] text-muted-foreground">Upper</label>
                  <input
                    type="number"
                    className="w-full p-2 border rounded bg-background text-sm"
                    value={config.upper_percentile ?? 95.0}
                    onChange={(e) => onChange({ ...config, upper_percentile: parseFloat(e.target.value) })}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Elliptic Envelope Params */}
          {config.method === 'elliptic_envelope' && (
            <div className="space-y-2">
              <label className="block text-sm font-medium">Contamination</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="0.5"
                className="w-full p-2 border rounded bg-background text-sm"
                value={config.contamination ?? 0.01}
                onChange={(e) => onChange({ ...config, contamination: parseFloat(e.target.value) })}
              />
              <p className="text-xs text-muted-foreground">Expected proportion of outliers in the dataset.</p>
            </div>
          )}
          
          {/* Feedback Section (Wide) */}
          {isWide && (
            <>
                <FeedbackSection />
                <RecommendationsSection />
            </>
          )}
        </div>
        
        {/* Right Column: Columns */}
        <div className={`flex flex-col h-full min-h-[200px] border rounded-md overflow-hidden ${isWide ? '' : 'shrink-0'}`}>
           <div className="p-2 border-b bg-muted/30 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Numeric Columns ({config.columns.length})</span>
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
                  <label key={col} className="flex items-center gap-2 text-sm hover:bg-accent/50 p-1.5 rounded cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={config.columns.includes(col)}
                      onChange={(e) => {
                        const newCols = e.target.checked
                          ? [...config.columns, col]
                          : config.columns.filter(c => c !== col);
                        onChange({ ...config, columns: newCols });
                      }}
                      className="rounded border-gray-300 text-primary focus:ring-primary"
                    />
                    <span className="truncate" title={col}>{col}</span>
                  </label>
                ))
            ) : (
              <div className="text-xs text-muted-foreground italic p-4 text-center">
                {isLoading ? 'Loading schema...' : (numericColumns.length === 0 ? 'No numeric columns found' : 'No matches found')}
              </div>
            )}
           </div>
        </div>

        {/* Feedback Section (Narrow) */}
        {!isWide && (
            <>
                <FeedbackSection />
                <RecommendationsSection />
            </>
        )}

      </div>
    </div>
  );
};

export const OutlierNode: NodeDefinition = {
  type: 'outlier',
  label: 'Outlier Removal',
  category: 'Preprocessing',
  description: 'Detect and remove or clip outliers.',
  icon: Scissors,
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Cleaned' }],
  settings: OutlierSettings,
  validate: (config: OutlierConfig) => {
    if (config.columns.length === 0) {
      return { isValid: false, message: 'Select at least one column.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'iqr',
    columns: [],
    multiplier: 1.5,
    threshold: 3.0,
    lower_percentile: 5.0,
    upper_percentile: 95.0,
    contamination: 0.01
  })
};
