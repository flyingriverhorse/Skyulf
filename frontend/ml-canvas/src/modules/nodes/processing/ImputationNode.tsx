import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { PaintBucket } from 'lucide-react';
import type { Recommendation } from '../../../core/api/client';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import type { ImputationConfig } from './imputation/types';
import { useImputationData } from './imputation/useImputationData';
import { ImputationMethodOptions } from './imputation/ImputationMethodOptions';
import { ImputationStatus, ImputationColumns, ImputationFeedback } from './imputation/ImputationPresentation';

const ImputationSettings: React.FC<{ config: ImputationConfig; onChange: (c: ImputationConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const data = useImputationData(nodeId);
  const { datasetId, isLoading, availableColumns, missingCounts, recommendations } = data;
  // Wide layouts keep feedback beside the method controls.
  const [containerRef, isWide] = useIsWideContainer();

  const handleApplyRecommendation = (rec: Recommendation) => {
    if (rec.target_columns.length > 0) {
      const newCols = Array.from(new Set([...config.columns, ...rec.target_columns]));
      // Apply strategy if recommended (assuming recommendation might contain params)
      // For now just columns
      onChange({ ...config, columns: newCols });
    }
  };

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      {/* Top Status Bar */}
      <ImputationStatus datasetId={datasetId} isLoading={isLoading} />

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

          <ImputationMethodOptions config={config} onChange={onChange} />

          {/* Feedback Section - Only show here if wide */}
          {isWide && <ImputationFeedback {...data} />}
        </div>

        {/* Right Column (Column Selection) */}
        <ImputationColumns config={config} onChange={onChange} availableColumns={availableColumns}
          missingCounts={missingCounts} isLoading={isLoading} isWide={isWide} />

        {/* Feedback Section - Show here if NOT wide (mobile/narrow) */}
        {!isWide && <ImputationFeedback {...data} />}
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
    if (config.columns.length === 0) return { isValid: false, field: 'columns', message: 'Select at least one column' };
    if (config.method === 'simple' && config.strategy === 'constant' && (config.fill_value === undefined || config.fill_value === '')) {
      return { isValid: false, field: 'fill_value', message: 'Fill value is required for Constant strategy' };
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
