import React, { useEffect } from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

export type DropMissingRowsSectionProps = {
  thresholdParameter: FeatureNodeParameter | null;
  dropIfAnyParameter: FeatureNodeParameter | null;
  dropIfAnyValue?: boolean;
  suppressWarnings?: boolean;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const DropMissingRowsSection: React.FC<DropMissingRowsSectionProps> = ({
  thresholdParameter,
  dropIfAnyParameter,
  dropIfAnyValue,
  suppressWarnings,
  renderParameterField,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (suppressWarnings) {
      return;
    }

    if (dropIfAnyValue) {
      onPendingConfigurationCleared?.();
      return;
    }

    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      const relevantDetails = details.filter((d) => d.label.toLowerCase().includes('rows'));

      if (relevantDetails.length > 0) {
        onPendingConfigurationWarning?.(relevantDetails);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared, dropIfAnyValue, suppressWarnings]);

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
