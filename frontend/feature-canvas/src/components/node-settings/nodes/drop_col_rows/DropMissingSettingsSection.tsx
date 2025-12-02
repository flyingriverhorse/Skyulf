import React, { useEffect } from 'react';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

type DropMissingSettingsSectionProps = {
  thresholdParameter: any;
  dropColumnParameter: any;
  renderParameterField: (param: any) => React.ReactNode;
  renderMultiSelectField: (param: any) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const DropMissingSettingsSection: React.FC<DropMissingSettingsSectionProps> = ({
  thresholdParameter,
  dropColumnParameter,
  renderParameterField,
  renderMultiSelectField,
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

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Missingness recommendations</h3>
      </div>
      {thresholdParameter && renderParameterField(thresholdParameter)}
      {renderMultiSelectField(dropColumnParameter)}
    </section>
  );
};
