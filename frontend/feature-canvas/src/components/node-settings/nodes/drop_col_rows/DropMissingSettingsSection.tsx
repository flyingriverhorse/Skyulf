import React, { useEffect } from 'react';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

type DropMissingSettingsSectionProps = {
  thresholdParameter: any;
  dropColumnParameter: any;
  dropIfAnyValue?: boolean;
  selectedColumns?: string[];
  renderParameterField: (param: any) => React.ReactNode;
  renderMultiSelectField: (param: any) => React.ReactNode;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const DropMissingSettingsSection: React.FC<DropMissingSettingsSectionProps> = ({
  thresholdParameter,
  dropColumnParameter,
  dropIfAnyValue,
  selectedColumns,
  renderParameterField,
  renderMultiSelectField,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      
      let relevantDetails = details;

      // If "Drop rows with any missing value" is enabled, suppress row warnings
      if (dropIfAnyValue) {
        relevantDetails = relevantDetails.filter((d) => !d.label.toLowerCase().includes('rows'));
      }

      // If specific columns are selected, suppress "threshold not configured" warnings for columns
      if (selectedColumns && selectedColumns.length > 0) {
        relevantDetails = relevantDetails.filter((d) => {
          const isThresholdWarning = d.label.toLowerCase().includes('threshold') || (d.reason && d.reason.toLowerCase().includes('threshold'));
          const isRowWarning = d.label.toLowerCase().includes('rows');
          // Filter out if it's a threshold warning AND NOT a row warning (i.e. it's a column threshold warning)
          return !(isThresholdWarning && !isRowWarning);
        });
      }

      if (relevantDetails.length > 0) {
        onPendingConfigurationWarning?.(relevantDetails);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared, dropIfAnyValue, selectedColumns]);

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
