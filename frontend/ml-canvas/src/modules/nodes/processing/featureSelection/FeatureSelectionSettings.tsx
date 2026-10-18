import React, { useState } from 'react';
import { useValidationReveal } from '../../../../components/shared/ValidationField';
import type { NodeSettingsProps } from '../../../../core/types/nodes';
import { useIsWideContainer } from '../../../../core/hooks/useIsWideContainer';
import { useFeatureSelectionData } from './useFeatureSelectionData';
import { SelectionControls, SelectionStatus } from './SelectionControls';
import { SelectionParameters } from './SelectionParameters';
import { FeatureSelectionFeedback } from './FeatureSelectionFeedback';
import type { FeatureSelectionConfig } from './types';

/** Load the complete selector editor only when its settings panel opens. */
export default function FeatureSelectionSettings({
  config,
  onChange,
  nodeId,
}: NodeSettingsProps<FeatureSelectionConfig>) {
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
}
