import React, { useState } from 'react';
import { NodeDefinition, ValidationResult } from '../../../core/types/nodes';
import { Activity } from 'lucide-react';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useRecommendations } from '../../../core/hooks/useRecommendations';
import { Recommendation } from '../../../core/api/client';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';

import { useResamplingData } from './resampling/useResamplingData';
import { GeneralSettings, MethodSettings, ResamplingRecommendations } from './resampling/ResamplingFields';

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
    const nodeResult = nodeId
        ? (executionResult?.node_results?.[nodeId] as
                | { metrics?: Record<string, unknown>; error?: string }
                | undefined)
        : undefined;

  if (!nodeResult) return null;

  // Helper to format metrics
  const renderMetrics = (metrics: Record<string, unknown>) => {
    // metrics is typed as Record<string, unknown>, so it's always truthy if it exists on the type.
    // Assuming strict null checks, if metrics is not optional, this check is redundant.
    // However, if it can be null/undefined, the check is valid.
    // Based on the error "Unnecessary conditional, value is always truthy", metrics is guaranteed.

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
  const id = React.useId();
  // Responsive layout: switch to a 2-column layout once the panel is wider than 400px.
  const [containerRef, isWide] = useIsWideContainer(400);
  const [showRecommendations, setShowRecommendations] = useState(true);

  const data = useResamplingData(nodeId, config, onChange);

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

            <GeneralSettings config={config} handleChange={handleChange} id={id} data={data} />
        </div>

        {/* Right Column (Advanced Params & Results) */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-700' : 'shrink-0 pt-4 border-t border-gray-100 dark:border-gray-700'}`}>

            <MethodSettings config={config} handleChange={handleChange} />

            <ResamplingRecommendations recommendations={recommendations} showRecommendations={showRecommendations}
              setShowRecommendations={setShowRecommendations} handleApplyRecommendation={handleApplyRecommendation} />

            {nodeId && <LastRunResults nodeId={nodeId} />}
        </div>
      </div>
    </div>
  );
};

// --- Node Definition ---

const validate = (data: ResamplingConfig): ValidationResult => {
  if (!data.target_column) {
    return { isValid: false, field: 'target_column', message: 'Target column is required for resampling.' };
  }

  if (data.type === 'oversampling') {
    if (['smote', 'adasyn', 'borderline_smote', 'svm_smote', 'kmeans_smote', 'smote_tomek'].includes(data.method)) {
       if ((data.k_neighbors ?? 5) < 1) {
           return { isValid: false, field: 'k_neighbors', message: 'k_neighbors must be at least 1.' };
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
    icon: Activity as unknown as React.FC<any>,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Balanced Data', type: 'dataset' }],
  settings: ResamplingSettings,
  bodyPreview: (config) => {
    const method = (config.method ?? 'smote').toUpperCase();
    const target = config.target_column;
    if (target) return `${method} → ${target}`;
    return method;
  },
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
