import { useValidationReveal } from '../../../components/shared/ValidationField';
import React, { useState } from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Filter } from 'lucide-react';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useFeatureSelectionData } from './featureSelection/useFeatureSelectionData';
import { SelectionControls, SelectionStatus } from './featureSelection/SelectionControls';
import { SelectionParameters } from './featureSelection/SelectionParameters';
import { FeatureSelectionFeedback } from './featureSelection/FeatureSelectionFeedback';
import type { FeatureSelectionConfig } from './featureSelection/types';

const FeatureSelectionSettings: React.FC<{ config: FeatureSelectionConfig; onChange: (c: FeatureSelectionConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const id = React.useId();
  const { upstreamDatasetId, upstreamTargetColumn, columns, isLoading, result } = useFeatureSelectionData({ config, onChange, nodeId });

  const [showInvalidTarget, setShowInvalidTarget] = useState(false);
  useValidationReveal((field) => {
    if (field === 'target_column') setShowInvalidTarget(true);
  });
  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();

  return (
    <div ref={containerRef} className={`flex flex-col h-full w-full bg-background ${isWide ? 'overflow-hidden' : 'overflow-y-auto'}`}>
      <SelectionStatus upstreamDatasetId={upstreamDatasetId} isLoading={isLoading} />

      {/* Main Content */}
      <div className={`flex-1 min-h-0 p-4 gap-4 ${isWide ? 'grid grid-cols-2' : 'flex flex-col'}`}>

        {/* Left Column: Method & Target */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pr-2' : 'shrink-0'}`}>
          <SelectionControls config={config} onChange={onChange} columns={columns} upstreamTargetColumn={upstreamTargetColumn} showInvalidTarget={showInvalidTarget} />

          {/* Feedback Section - Show here if wide (Left Column) */}
          {isWide && <FeatureSelectionFeedback config={config} result={result} />}
        </div>

        {/* Right Column: Parameters */}
        <div className={`space-y-4 ${isWide ? 'overflow-y-auto pl-2 border-l' : 'shrink-0 border-t pt-4'}`}>
          <h4 className="text-xs font-semibold uppercase text-muted-foreground mb-2">Parameters</h4>

          <SelectionParameters config={config} onChange={onChange} id={id} />

          {/* Feedback Section - Show here if NOT wide (Mobile/Narrow) */}
          {!isWide && <FeatureSelectionFeedback config={config} result={result} />}
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
  bodyPreview: (config) => {
    const method = config.method ?? 'variance_threshold';
    if ((method === 'select_k_best' || method === 'rfe') && config.k != null) return `${method} · k=${config.k}`;
    if (method === 'select_percentile' && config.percentile != null) return `${method} · ${config.percentile}%`;
    if (method === 'variance_threshold' && config.threshold != null) return `${method} · σ>${config.threshold}`;
    return method;
  },
  validate: (config) => {
    if (config.method !== 'variance_threshold' && !config.target_column) {
      return { isValid: false, field: 'target_column', message: 'Target column is required for this method.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'select_k_best',
    k: 10,
  }),
};
