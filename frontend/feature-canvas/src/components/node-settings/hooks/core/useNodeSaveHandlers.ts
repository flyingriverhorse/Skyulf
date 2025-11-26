import { useCallback, type Dispatch, type SetStateAction } from 'react';
import { triggerFullDatasetExecution } from '../../../../api';
import { normalizeBinningConfigValue } from '../../nodes/binning/binningSettings';
import { normalizeScalingConfigValue } from '../../nodes/scaling/scalingSettings';
import { cloneConfig } from '../../utils/configParsers';
import type { CatalogFlagMap } from './useCatalogFlags';

export type UseNodeSaveHandlersArgs = {
  configState: Record<string, any>;
  catalogFlags: CatalogFlagMap;
  nodeId: string;
  onUpdateConfig: (nodeId: string, config: Record<string, any>) => void;
  onClose: () => void;
  sourceId?: string | null;
  graphSnapshot?: { nodes: any[]; edges: any[] } | null;
  onUpdateNodeData?: (nodeId: string, dataUpdates: Record<string, any>) => void;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
  canResetNode: boolean;
  defaultConfigTemplate?: Record<string, any> | null;
  onResetConfig?: (nodeId: string, config?: Record<string, any> | null) => void;
};

export const useNodeSaveHandlers = ({
  configState,
  catalogFlags,
  nodeId,
  onUpdateConfig,
  onClose,
  sourceId,
  graphSnapshot,
  onUpdateNodeData,
  setConfigState,
  canResetNode,
  defaultConfigTemplate,
  onResetConfig,
}: UseNodeSaveHandlersArgs) => {
  const { isBinningNode, isScalingNode, isInspectionNode } = catalogFlags;

  const handleSave = useCallback(
    (options?: { closeModal?: boolean }) => {
      if (!nodeId) {
        return;
      }

      let payload = cloneConfig(configState);

      if (isBinningNode) {
        const normalized = normalizeBinningConfigValue(payload);
        payload = {
          ...payload,
          strategy: normalized.strategy,
          columns: normalized.columns,
          equal_width_bins: normalized.equalWidthBins,
          equal_frequency_bins: normalized.equalFrequencyBins,
          include_lowest: normalized.includeLowest,
          precision: normalized.precision,
          duplicates: normalized.duplicates,
          output_suffix: normalized.outputSuffix,
          drop_original: normalized.dropOriginal,
          label_format: normalized.labelFormat,
          missing_strategy: normalized.missingStrategy,
        };

        if (normalized.missingStrategy === 'label') {
          payload.missing_label = normalized.missingLabel;
        } else if (Object.prototype.hasOwnProperty.call(payload, 'missing_label')) {
          delete payload.missing_label;
        }

        if (Object.keys(normalized.customBins).length) {
          payload.custom_bins = normalized.customBins;
        } else if (Object.prototype.hasOwnProperty.call(payload, 'custom_bins')) {
          delete payload.custom_bins;
        }

        if (Object.keys(normalized.customLabels).length) {
          payload.custom_labels = normalized.customLabels;
        } else if (Object.prototype.hasOwnProperty.call(payload, 'custom_labels')) {
          delete payload.custom_labels;
        }
      }

      if (isScalingNode) {
        const normalized = normalizeScalingConfigValue(payload);
        payload = {
          ...payload,
          columns: normalized.columns,
          default_method: normalized.defaultMethod,
          auto_detect: normalized.autoDetect,
          skipped_columns: normalized.skippedColumns,
        };

        if (Object.keys(normalized.columnMethods).length > 0) {
          payload.column_methods = normalized.columnMethods;
        } else if (Object.prototype.hasOwnProperty.call(payload, 'column_methods')) {
          delete payload.column_methods;
        }
      }

      onUpdateConfig(nodeId, payload);
      if (options?.closeModal !== false) {
        onClose();
      }

      if (sourceId && graphSnapshot && !isInspectionNode) {
        onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: 'loading' });
        triggerFullDatasetExecution({
          dataset_source_id: sourceId,
          graph: {
            nodes: graphSnapshot.nodes || [],
            edges: graphSnapshot.edges || [],
          },
          target_node_id: nodeId,
        })
          .then(() => {
            onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: 'success' });
          })
          .catch((error: unknown) => {
            onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: 'error' });
            console.warn('Background full dataset execution failed:', error);
          });
      }
    },
    [
      configState,
      graphSnapshot,
      isBinningNode,
      isInspectionNode,
      isScalingNode,
      nodeId,
      onClose,
      onUpdateConfig,
      onUpdateNodeData,
      sourceId,
    ]
  );

  const handleResetNode = useCallback(() => {
    if (!canResetNode || !nodeId) {
      return;
    }
    const template = cloneConfig(defaultConfigTemplate ?? {});
    setConfigState(template);
    onResetConfig?.(nodeId, template);
  }, [canResetNode, defaultConfigTemplate, nodeId, onResetConfig, setConfigState]);

  return {
    handleSave,
    handleResetNode,
  };
};
