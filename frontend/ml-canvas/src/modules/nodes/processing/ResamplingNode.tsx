import React, { useState, useRef, useEffect } from 'react';
import { NodeDefinition, ValidationResult } from '../../../core/types/nodes';
import { Activity, ChevronDown, ChevronUp } from 'lucide-react';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { Recommendation, ColumnProfile } from '../../../core/api/client';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';

// --- Types ---

interface ResamplingConfig {
  type: 'oversampling' | 'undersampling';
  method: string;
  target_column: string;
  sampling_strategy: string;
  random_state: number;
  // Oversampling
  k_neighbors?: number;
  m_neighbors?: number;
  kind?: string;
  out_step?: number;
  cluster_balance_threshold?: number;
  density_exponent?: string;
  // Undersampling
  replacement?: boolean;
  version?: number;
  n_neighbors?: number;
  kind_sel?: string;
}

// --- Components ---

const LastRunResults: React.FC<{ nodeId: string }> = ({ nodeId }) => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const nodeResult = nodeId ? executionResult?.node_results[nodeId] : null;
  
  if (!nodeResult) return null;

  // Helper to format metrics
  const renderMetrics = (metrics: Record<string, unknown>) => {
    if (!metrics) return null;
    return (
      <div className="space-y-1">
        {Object.entries(metrics).map(([key, value]) => {
          if (typeof value === 'object' && value !== null) {
             return (
               <div key={key} className="mt-1">
                 <span className="font-medium text-gray-700 dark:text-gray-300">{key.replace(/_/g, ' ')}:</span>
                 <div className="pl-2 border-l-2 border-gray-200 dark:border-gray-700 ml-1">
                    {renderMetrics(value as Record<string, unknown>)}
                 </div>
               </div>
             );
          }
          return (
            <div key={key} className="flex justify-between text-[10px]">
              <span className="text-gray-600 dark:text-gray-400 capitalize">{key.replace(/_/g, ' ')}:</span>
              <span className="font-mono text-gray-900 dark:text-gray-100">{String(value)}</span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="mt-6 p-4 border rounded-md bg-gray-50 dark:bg-gray-800 dark:border-gray-700">
      <h4 className="text-sm font-medium mb-2 text-gray-900 dark:text-gray-100">Last Run Results</h4>
      <div className="text-xs space-y-1 text-gray-600 dark:text-gray-300">
        {nodeResult.metrics && (
            <div className="mt-2">
                {renderMetrics(nodeResult.metrics)}
            </div>
        )}
        {nodeResult.error && (
            <div className="mt-2 text-red-500 break-words">
                Error: {nodeResult.error}
            </div>
        )}
        {!nodeResult.metrics && !nodeResult.error && (
            <div className="italic text-gray-500">No metrics available</div>
        )}
      </div>
    </div>
  );
};

const ResamplingSettings: React.FC<{ config: ResamplingConfig; onChange: (c: ResamplingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  // Responsive Layout Logic
  const containerRef = useRef<HTMLDivElement>(null);
  const [isWide, setIsWide] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(true);

  // Upstream Data for Target Column Suggestion
  const upstreamData = useUpstreamData(nodeId || '') as Record<string, unknown>[];
  const datasetId = upstreamData.find((d: Record<string, unknown>) => d.datasetId)?.datasetId as string | undefined;
  const { data: schema } = useDatasetSchema(datasetId);
  
  // Try to find a target column from upstream nodes configuration
  const upstreamTarget = upstreamData.find((d: Record<string, unknown>) => (d.config as Record<string, unknown>)?.target_column);
  const targetColumn = upstreamTarget ? ((upstreamTarget.config as Record<string, unknown>).target_column as string) : undefined;

  // Auto-fill target column if empty and available in schema or upstream
  useEffect(() => {
      if (!config.target_column) {
          if (upstreamTarget) {
               onChange({ ...config, target_column: upstreamTarget });
          } else if (schema?.columns) {
               // Simple heuristic: check for 'target' or 'class' or column_type
               const potentialTarget = Object.values(schema.columns).find((c: ColumnProfile) => 
                   c.name.toLowerCase() === 'target' || 
                   c.name.toLowerCase() === 'class' ||
                   c.column_type === 'target'
               );
               if (potentialTarget) {
                   onChange({ ...config, target_column: potentialTarget.name });
               }
          }
      }
  }, [schema, upstreamTarget, config.target_column]);

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

  // Recommendations
  const recommendations = useRecommendations(nodeId || '', {
    types: ['resampling'],
    suggestedNodeTypes: ['ResamplingNode'],
  });

  const handleApplyRecommendation = (rec: Recommendation) => {
    if (rec.suggested_params) {
        onChange({ ...config, ...rec.suggested_params });
    }
  };

  const handleChange = (key: keyof ResamplingConfig, value: unknown) => {
    const newConfig = { ...config, [key]: value };
    
    // Reset method defaults when type changes
    if (key === 'type') {
        if (value === 'oversampling') {
            newConfig.method = 'smote';
        } else {
            newConfig.method = 'random_under_sampling';
        }
    }
    
    onChange(newConfig);
  };

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-white dark:bg-gray-900 ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>
        
        {/* Left Column (Main Settings) */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
            
            <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Resampling Type</label>
                <select 
                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    value={config.type}
                    onChange={(e) => { handleChange('type', e.target.value as ResamplingConfig['type']); }}
                >
                    <option value="oversampling">Oversampling (Minority)</option>
                    <option value="undersampling">Undersampling (Majority)</option>
                </select>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Method</label>
                <select 
                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    value={config.method}
                    onChange={(e) => { handleChange('method', e.target.value); }}
                >
                    {config.type === 'oversampling' ? (
                        <>
                            <option value="smote">SMOTE</option>
                            <option value="adasyn">ADASYN</option>
                            <option value="borderline_smote">Borderline SMOTE</option>
                            <option value="svm_smote">SVM SMOTE</option>
                            <option value="kmeans_smote">KMeans SMOTE</option>
                            <option value="smote_tomek">SMOTE + Tomek</option>
                        </>
                    ) : (
                        <>
                            <option value="random_under_sampling">Random Under Sampling</option>
                            <option value="nearmiss">NearMiss</option>
                            <option value="tomek_links">Tomek Links</option>
                            <option value="edited_nearest_neighbours">Edited Nearest Neighbours</option>
                        </>
                    )}
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                    {config.type === 'oversampling' 
                        ? 'Generate synthetic samples for the minority class.' 
                        : 'Remove samples from the majority class.'}
                </p>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Target Column</label>
                <div className="relative">
                    <input 
                        type="text"
                        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        placeholder="e.g., target"
                        value={config.target_column}
                        onChange={(e) => { handleChange('target_column', e.target.value); }}
                        list="target-column-suggestions"
                    />
                    {schema?.columns && (
                        <datalist id="target-column-suggestions">
                            {Object.values(schema.columns).map((col: ColumnProfile) => (
                                <option key={col.name} value={col.name} />
                            ))}
                        </datalist>
                    )}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                    {upstreamTarget ? 'Auto-detected from upstream node.' : 'The column containing the class labels.'}
                </p>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Sampling Strategy</label>
                <select 
                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    value={config.sampling_strategy}
                    onChange={(e) => { handleChange('sampling_strategy', e.target.value); }}
                >
                    <option value="auto">Auto (Resample all classes but majority)</option>
                    <option value="minority">Minority (Resample only minority class)</option>
                    <option value="not minority">Not Minority (Resample all but minority)</option>
                    <option value="not majority">Not Majority (Resample all but majority)</option>
                    <option value="all">All (Resample all classes)</option>
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400">Defines which classes to resample.</p>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Random State</label>
                <input 
                    type="number"
                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    value={config.random_state}
                    onChange={(e) => { handleChange('random_state', parseInt(e.target.value)); }}
                />
            </div>

        </div>

        {/* Right Column (Advanced Params & Results) */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>
            
            {/* Method Specific Params */}
            {config.type === 'oversampling' && (
                <>
                    {['smote', 'adasyn', 'borderline_smote', 'svm_smote', 'kmeans_smote'].includes(config.method) && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">k Neighbors</label>
                            <input 
                                type="number"
                                className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                value={config.k_neighbors ?? 5}
                                onChange={(e) => { handleChange('k_neighbors', parseInt(e.target.value)); }}
                            />
                        </div>
                    )}
                    
                    {['borderline_smote', 'svm_smote'].includes(config.method) && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">m Neighbors</label>
                            <input 
                                type="number"
                                className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                value={config.m_neighbors ?? 10}
                                onChange={(e) => { handleChange('m_neighbors', parseInt(e.target.value)); }}
                            />
                        </div>
                    )}

                    {config.method === 'borderline_smote' && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Kind</label>
                            <select 
                                className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                value={config.kind ?? 'borderline-1'}
                                onChange={(e) => { handleChange('kind', e.target.value); }}
                            >
                                <option value="borderline-1">Borderline-1</option>
                                <option value="borderline-2">Borderline-2</option>
                            </select>
                        </div>
                    )}

                    {config.method === 'svm_smote' && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Out Step</label>
                            <input 
                                type="number"
                                step="0.1"
                                className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                value={config.out_step ?? 0.5}
                                onChange={(e) => { handleChange('out_step', parseFloat(e.target.value)); }}
                            />
                        </div>
                    )}

                    {config.method === 'kmeans_smote' && (
                        <>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Cluster Balance Threshold</label>
                                <input 
                                    type="number"
                                    step="0.1"
                                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                    value={config.cluster_balance_threshold ?? 0.1}
                                    onChange={(e) => { handleChange('cluster_balance_threshold', parseFloat(e.target.value)); }}
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Density Exponent</label>
                                <input 
                                    type="text"
                                    className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                    value={config.density_exponent ?? 'auto'}
                                    onChange={(e) => { handleChange('density_exponent', e.target.value); }}
                                />
                            </div>
                        </>
                    )}
                </>
            )}

            {config.type === 'undersampling' && config.method === 'random_under_sampling' && (
                <div className="flex items-center justify-between p-3 border rounded-md bg-gray-50 dark:bg-gray-800 dark:border-gray-700">
                    <div>
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block">Replacement</label>
                        <span className="text-xs text-gray-500 dark:text-gray-400">Sample with replacement</span>
                    </div>
                    <input 
                        type="checkbox"
                        className="h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
                        checked={config.replacement ?? false}
                        onChange={(e) => { handleChange('replacement', e.target.checked); }}
                    />
                </div>
            )}

            {config.type === 'undersampling' && config.method === 'nearmiss' && (
                <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Version</label>
                    <select 
                        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={config.version ?? 1}
                        onChange={(e) => { handleChange('version', parseInt(e.target.value)); }}
                    >
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                    </select>
                </div>
            )}

            {config.type === 'undersampling' && config.method === 'edited_nearest_neighbours' && (
                <>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">n Neighbors</label>
                        <input 
                            type="number"
                            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            value={config.n_neighbors ?? 3}
                            onChange={(e) => { handleChange('n_neighbors', parseInt(e.target.value)); }}
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Selection Kind</label>
                        <select 
                            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                            value={config.kind_sel ?? 'all'}
                            onChange={(e) => { handleChange('kind_sel', e.target.value); }}
                        >
                            <option value="all">All</option>
                            <option value="mode">Mode</option>
                        </select>
                    </div>
                </>
            )}

            {recommendations.length > 0 && (
                <div className="mt-0 border rounded-md overflow-hidden border-gray-200 dark:border-gray-700">
                    <button 
                        className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        onClick={() => { setShowRecommendations(!showRecommendations); }}
                    >
                        <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            Recommendations ({recommendations.length})
                        </span>
                        {showRecommendations ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    
                    {showRecommendations && (
                        <div className="p-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
                            <RecommendationsPanel 
                                recommendations={recommendations} 
                                onApply={handleApplyRecommendation}
                            />
                        </div>
                    )}
                </div>
            )}

            {nodeId && <LastRunResults nodeId={nodeId} />}
        </div>
      </div>
    </div>
  );
};

// --- Node Definition ---

const validate = (data: ResamplingConfig): ValidationResult => {
  if (!data.target_column) {
    return { isValid: false, message: 'Target column is required for resampling.' };
  }
  
  if (data.type === 'oversampling') {
    if (['smote', 'adasyn', 'borderline_smote', 'svm_smote', 'kmeans_smote'].includes(data.method)) {
       if ((data.k_neighbors ?? 5) < 1) {
           return { isValid: false, message: 'k_neighbors must be at least 1.' };
       }
    }
  }

  return { isValid: true };
};

export const ResamplingNode: NodeDefinition<ResamplingConfig> = {
  type: 'ResamplingNode',
  label: 'Resampling',
  category: 'Preprocessing',
  description: 'Balance dataset classes using oversampling or undersampling techniques.',
  icon: Activity,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Balanced Data', type: 'dataset' }],
  settings: ResamplingSettings,
  validate: validate,
  getDefaultConfig: () => ({
    type: 'oversampling',
    method: 'smote',
    target_column: '',
    sampling_strategy: 'auto',
    random_state: 42,
    k_neighbors: 5,
    replacement: false,
    version: 1,
    n_neighbors: 3,
    kind_sel: 'all'
  })
};
