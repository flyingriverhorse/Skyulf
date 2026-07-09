import React, { useMemo } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { PaintBucket } from 'lucide-react';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useUpstreamDroppedColumns } from '../../../core/hooks/useUpstreamDroppedColumns';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { Recommendation } from '../../../core/api/client';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { parseIntSafe } from '../../../core/utils/numberInput';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';

interface ImputationConfig {
  columns: string[];
  method: 'simple' | 'knn' | 'iterative';

  // Simple Imputer
  strategy: 'mean' | 'median' | 'most_frequent' | 'constant';
  fill_value?: string | number | undefined;

  // KNN Imputer
  n_neighbors?: number | undefined;
  weights?: 'uniform' | 'distance' | undefined;

  // Iterative Imputer
  max_iter?: number | undefined;
  estimator?: 'bayesian_ridge' | 'decision_tree' | 'extra_trees' | 'knn' | undefined;
  random_state?: number | undefined;
}

const ImputationSettings: React.FC<{ config: ImputationConfig; onChange: (c: ImputationConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const upstreamData = useUpstreamData(nodeId || '');
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId as string | undefined;
  const { data: schema, isLoading } = useDatasetSchema(datasetId);
  const droppedUpstream = useUpstreamDroppedColumns(nodeId);
  const availableColumns = useMemo(
    () => schema ? Object.values(schema.columns).map(c => c.name).filter(n => !droppedUpstream.has(n)) : [],
    [schema, droppedUpstream]
  );

  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  const metrics: Record<string, unknown> | null =
    nodeResult?.metrics && typeof nodeResult.metrics === 'object'
      ? (nodeResult.metrics as Record<string, unknown>)
      : null;
  const fillValues: Record<string, unknown> | null =
    metrics?.fill_values && typeof metrics.fill_values === 'object'
      ? (metrics.fill_values as Record<string, unknown>)
      : null;
  const missingCounts: Record<string, unknown> | null =
    metrics?.missing_counts && typeof metrics.missing_counts === 'object'
      ? (metrics.missing_counts as Record<string, unknown>)
      : null;

  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();

  const recommendations = useRecommendations(nodeId || '', {
    types: ['imputation'],
    suggestedNodeTypes: ['SimpleImputer', 'KNNImputer', 'IterativeImputer'],
    scope: 'column'
  });

  const handleApplyRecommendation = (rec: Recommendation) => {
    if (rec.target_columns.length > 0) {
      const newCols = Array.from(new Set([...config.columns, ...rec.target_columns]));
      // Apply strategy if recommended (assuming recommendation might contain params)
      // For now just columns
      onChange({ ...config, columns: newCols });
    }
  };

  const renderFeedback = () => (
    metrics ? (
      <div className="mt-4 p-3 bg-muted/50 rounded border text-xs">
        <div className="font-medium text-muted-foreground mb-2">Execution Feedback</div>

        {fillValues && (
          <div className="space-y-1">
            <span className="text-muted-foreground block mb-1 font-medium">Imputed Values:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
              {Object.entries(fillValues).map(([col, val]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{typeof val === 'number' ? val.toFixed(4) : String(val)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {missingCounts && (
          <div className="space-y-1 mt-2">
            <span className="text-muted-foreground block mb-1 font-medium">Missing Values Filled:</span>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 max-h-32 overflow-y-auto">
              {Object.entries(missingCounts).map(([col, count]) => (
                <div key={col} className="flex justify-between text-[10px]">
                  <span className="truncate max-w-[100px]" title={col}>{col}:</span>
                  <span className="font-mono">{String(count)}</span>
                </div>
              ))}
            </div>
            {metrics.total_missing !== undefined && (
               <div className="text-[10px] text-muted-foreground mt-1 pt-1 border-t">
                  Total Filled: <span className="font-mono font-medium">{String(metrics.total_missing)}</span>
               </div>
            )}
          </div>
        )}

        {/* Generic success message if no specific metrics but successful */}
        {!fillValues && !missingCounts && nodeResult?.status === 'success' && (
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
            <span className="block text-sm font-medium mb-1">Imputation Method</span>
            <select
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.method || 'simple'}
              onChange={(e) => onChange({ ...config, method: e.target.value as ImputationConfig['method'] })}
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
                <span className="block text-sm font-medium mb-1">Strategy</span>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.strategy}
                  onChange={(e) => onChange({ ...config, strategy: e.target.value as ImputationConfig['strategy'] })}
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
                  <span className="block text-sm font-medium mb-1">Fill Value</span>
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
                <span className="block text-sm font-medium mb-1">Number of Neighbors</span>
                <input
                  type="number"
                  min="1"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.n_neighbors || 5}
                  onChange={(e) => onChange({ ...config, n_neighbors: parseIntSafe(e.target.value, config.n_neighbors) })}
                />
              </div>
              <div>
                <span className="block text-sm font-medium mb-1">Weights</span>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.weights || 'uniform'}
                  onChange={(e) => onChange({ ...config, weights: e.target.value as ImputationConfig['weights'] })}
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
                <span className="block text-sm font-medium mb-1">Max Iterations</span>
                <input
                  type="number"
                  min="1"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.max_iter || 10}
                  onChange={(e) => onChange({ ...config, max_iter: parseIntSafe(e.target.value, config.max_iter) })}
                />
              </div>
              <div>
                <span className="block text-sm font-medium mb-1">Estimator</span>
                <select
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.estimator || 'bayesian_ridge'}
                  onChange={(e) => onChange({ ...config, estimator: e.target.value as ImputationConfig['estimator'] })}
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
                <span className="block text-sm font-medium mb-1">Random State</span>
                <input
                  type="number"
                  className="w-full p-2 border rounded bg-background text-sm"
                  value={config.random_state ?? 0}
                  onChange={(e) => onChange({ ...config, random_state: parseIntSafe(e.target.value, config.random_state) })}
                />
              </div>
            </>
          )}

          {/* Feedback Section - Only show here if wide */}
          {isWide && renderFeedback()}
        </div>

        {/* Right Column (Column Selection) */}
        <div className={`flex flex-col overflow-hidden ${isWide ? 'min-h-0 flex-1' : 'shrink-0'}`}>
          <ColumnMultiSelect
            columns={availableColumns}
            selected={config.columns}
            onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
            label="Target Columns"
            variant="panel"
            isLoading={isLoading}
            fillHeight={isWide}
            renderItemBadge={(col) =>
              missingCounts && missingCounts[col] !== undefined ? (
                <span
                  className="text-[10px] text-muted-foreground font-mono shrink-0 bg-muted px-1.5 py-0.5 rounded"
                  title={`${String(missingCounts[col])} missing values filled`}
                >
                  {String(missingCounts[col])}
                </span>
              ) : null
            }
          />
        </div>

        {/* Feedback Section - Show here if NOT wide (mobile/narrow) */}
        {!isWide && renderFeedback()}
      </div>
    </div>
  );
};

export const ImputationNode: NodeDefinition<ImputationConfig> = {
  type: 'imputation_node',
  label: 'Imputation',
  category: 'Preprocessing',
  description: 'Fill missing values.',
  icon: PaintBucket,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Data', type: 'dataset' }],
  settings: ImputationSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const strat = config.strategy ?? config.method ?? 'mean';
    if (cols === 0) return null;
    return `${strat} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: (config) => {
    if (config.columns.length === 0) return { isValid: false, message: 'Select at least one column' };
    if (config.method === 'simple' && config.strategy === 'constant' && (config.fill_value === undefined || config.fill_value === '')) {
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
