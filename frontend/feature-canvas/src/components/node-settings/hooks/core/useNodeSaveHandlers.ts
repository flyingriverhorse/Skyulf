import { useCallback, useRef, type Dispatch, type SetStateAction } from 'react';
import { fetchFullExecutionStatus, triggerFullDatasetExecution, type FullExecutionSignal } from '../../../../api';
import { normalizeBinningConfigValue } from '../../nodes/binning/binningSettings';
import { normalizeScalingConfigValue } from '../../nodes/scaling/scalingSettings';
import { cloneConfig } from '../../utils/configParsers';
import {
  extractPendingConfigurationDetails,
  type PendingConfigurationDetail,
} from '../../utils/pendingConfiguration';
import { PENDING_CONFIRMATION_FLAG } from '../../../../canvas/services/configSanitizer';
import type { CatalogFlagMap } from './useCatalogFlags';

type BackgroundExecutionStatus = 'idle' | 'loading' | 'success' | 'error';

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
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
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
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}: UseNodeSaveHandlersArgs) => {
  const { isBinningNode, isScalingNode, isInspectionNode } = catalogFlags;
  const activePollCancelRef = useRef<(() => void) | null>(null);

  const mapSignalToBackgroundStatus = useCallback((signal?: FullExecutionSignal | null): BackgroundExecutionStatus => {
    if (!signal) {
      return 'idle';
    }
    const token = (signal.job_status ?? signal.status ?? '').toLowerCase();
    if (!token) {
      return 'idle';
    }
    if (token === 'succeeded') {
      return 'success';
    }
    if (token === 'failed' || token === 'skipped' || token === 'cancelled') {
      return 'error';
    }
    if (token === 'running' || token === 'queued' || token === 'deferred') {
      return 'loading';
    }
    return 'idle';
  }, []);

  const startBackgroundStatusPolling = useCallback(
    (signal?: FullExecutionSignal | null) => {
      activePollCancelRef.current?.();

      if (!signal) {
        onPendingConfigurationCleared?.();
        return;
      }
      const datasetSource = sourceId;
      if (!datasetSource || !nodeId) {
        return;
      }
      const pendingDetails = extractPendingConfigurationDetails(signal);
      if (pendingDetails.length) {
        onPendingConfigurationWarning?.(pendingDetails);
      } else {
        onPendingConfigurationCleared?.();
      }
      const datasetSourceId = datasetSource as string;
      const initialStatus = mapSignalToBackgroundStatus(signal);
      if (initialStatus !== 'loading') {
        return;
      }
      const jobId = signal.job_id;
      if (!jobId) {
        return;
      }

      let cancelled = false;
      const poll = async (pollAfterSeconds?: number | null) => {
        const delayMs = typeof pollAfterSeconds === 'number' && pollAfterSeconds > 0 ? pollAfterSeconds * 1000 : 5000;
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        if (cancelled) {
          return;
        }
        try {
          const updatedSignal = await fetchFullExecutionStatus(datasetSourceId, jobId);
          const nextStatus = mapSignalToBackgroundStatus(updatedSignal);
          onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: nextStatus });
          const pendingFromPoll = extractPendingConfigurationDetails(updatedSignal);
          if (pendingFromPoll.length) {
            onPendingConfigurationWarning?.(pendingFromPoll);
          } else {
            onPendingConfigurationCleared?.();
          }
          if (nextStatus === 'loading') {
            poll(updatedSignal?.poll_after_seconds ?? null);
          }
        } catch (error) {
          console.warn('Failed to poll full dataset execution status:', error);
          onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: 'error' });
        }
      };

      poll(signal.poll_after_seconds ?? null);

      activePollCancelRef.current = () => {
        cancelled = true;
      };
    },
    [
      mapSignalToBackgroundStatus,
      nodeId,
      onPendingConfigurationCleared,
      onPendingConfigurationWarning,
      onUpdateNodeData,
      sourceId,
    ]
  );

  const mergeGraphSnapshotWithConfig = useCallback(
    (
      snapshot: { nodes: any[]; edges: any[] } | null | undefined,
      targetNodeId: string,
      nextConfig: Record<string, any>
    ) => {
      if (!snapshot) {
        return null;
      }

      const sanitizedConfig = cloneConfig(nextConfig);
      if (sanitizedConfig && typeof sanitizedConfig === 'object') {
        delete (sanitizedConfig as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
      }

      const patchedNodes = Array.isArray(snapshot.nodes)
        ? snapshot.nodes.map((node) => {
            if (!node || node.id !== targetNodeId) {
              return node;
            }

            const nextData = {
              ...(node.data ?? {}),
              config: sanitizedConfig,
              isConfigured: true,
            };

            return {
              ...node,
              data: nextData,
            };
          })
        : snapshot.nodes;

      return {
        nodes: patchedNodes,
        edges: Array.isArray(snapshot.edges) ? [...snapshot.edges] : snapshot.edges,
      };
    },
    []
  );

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

      const updatedGraphSnapshot = mergeGraphSnapshotWithConfig(graphSnapshot, nodeId, payload);

      if (sourceId && updatedGraphSnapshot && !isInspectionNode) {
        onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: 'loading' });
        triggerFullDatasetExecution({
          dataset_source_id: sourceId,
          graph: {
            nodes: updatedGraphSnapshot.nodes || [],
            edges: updatedGraphSnapshot.edges || [],
          },
          target_node_id: nodeId,
        })
          .then((response) => {
            const fullExecutionSignal = response?.signals?.full_execution;
            const nextStatus = mapSignalToBackgroundStatus(fullExecutionSignal);
            onUpdateNodeData?.(nodeId, { backgroundExecutionStatus: nextStatus });
            const pendingDetails = extractPendingConfigurationDetails(fullExecutionSignal);
            if (pendingDetails.length) {
              onPendingConfigurationWarning?.(pendingDetails);
            } else {
              onPendingConfigurationCleared?.();
            }
            startBackgroundStatusPolling(fullExecutionSignal);
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
      mergeGraphSnapshotWithConfig,
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
