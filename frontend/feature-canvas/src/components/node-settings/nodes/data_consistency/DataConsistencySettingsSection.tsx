import React, { useEffect } from 'react';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

type DataConsistencySettingsSectionProps = {
  parameters: any[];
  renderParameterField: (param: any) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const DataConsistencySettingsSection: React.FC<DataConsistencySettingsSectionProps> = ({
  parameters,
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

  if (parameters.length === 0) {
    return null;
  }

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Data consistency settings</h3>
      </div>
      <div className="canvas-modal__parameter-list">
        {parameters.map((parameter) => renderParameterField(parameter))}
      </div>
    </section>
  );
};
