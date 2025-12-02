import React, { useEffect } from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

export type DropMissingRowsSectionProps = {
  thresholdParameter: FeatureNodeParameter | null;
  dropIfAnyParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const DropMissingRowsSection: React.FC<DropMissingRowsSectionProps> = ({
  thresholdParameter,
  dropIfAnyParameter,
  renderParameterField,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      if (details.length > 0) {
        onPendingConfigurationWarning?.(details);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared]);

  if (!thresholdParameter && !dropIfAnyParameter) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Row missingness</h3>
      </div>
      {thresholdParameter && renderParameterField(thresholdParameter)}
      {dropIfAnyParameter && renderParameterField(dropIfAnyParameter)}
    </section>
  );
};
