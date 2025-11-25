import React, { useMemo } from 'react';
import type {
  FeatureNodeParameter,
  FeatureTargetSplitNodeSignal,
  PipelinePreviewColumnStat,
} from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';

export type FeatureTargetSplitConfig = {
  targetColumn: string;
  featureColumns: string[];
};

type FeatureTargetSplitSectionProps = {
  nodeId: string;
  sourceId?: string | null;
  hasReachableSource: boolean;
  previewState: PreviewState;
  onRefreshPreview: () => void;
  config: FeatureTargetSplitConfig | null;
  targetColumnParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  formatMissingPercentage: (value?: number | null) => string;
};

const summarizeList = (values: string[], limit = 5): string => {
  if (!values.length) {
    return '';
  }
  if (values.length <= limit) {
    return values.join(', ');
  }
  const preview = values.slice(0, limit).join(', ');
  return `${preview}, ...`;
};

const findSplitSummary = (steps: unknown[]): string | null => {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const entry = steps[index];
    if (typeof entry !== 'string') {
      continue;
    }
    const lowered = entry.toLowerCase();
    if (lowered.includes('feature/target split')) {
      return entry;
    }
  }
  return null;
};

const resolveSignal = (
  signals: FeatureTargetSplitNodeSignal[],
  nodeId: string,
): FeatureTargetSplitNodeSignal | null => {
  if (!signals.length) {
    return null;
  }
  const matching = signals.filter((signal) => !signal?.node_id || signal.node_id === nodeId);
  if (matching.length) {
    return matching[matching.length - 1];
  }
  return signals[signals.length - 1];
};

export const FeatureTargetSplitSection: React.FC<FeatureTargetSplitSectionProps> = ({
  nodeId,
  sourceId,
  hasReachableSource,
  previewState,
  onRefreshPreview,
  config,
  targetColumnParameter,
  renderParameterField,
  formatMetricValue,
  formatMissingPercentage,
}) => {
  const preview = previewState.data;
  const previewStatus = previewState.status;
  const appliedSteps = Array.isArray(preview?.applied_steps) ? preview.applied_steps : [];
  const previewColumns = Array.isArray(preview?.columns) ? (preview?.columns as string[]) : [];

  const rawSignals = Array.isArray(preview?.signals?.feature_target_split)
    ? (preview?.signals?.feature_target_split as FeatureTargetSplitNodeSignal[])
    : [];
  const activeSignal = resolveSignal(rawSignals, nodeId);

  const configuredTarget = config?.targetColumn ?? '';
  const resolvedTarget = activeSignal?.target_column ?? configuredTarget;
  const configuredFeatures = config?.featureColumns ?? [];

  const columnStats: PipelinePreviewColumnStat[] = useMemo(
    () => (Array.isArray(preview?.column_stats) ? (preview?.column_stats as PipelinePreviewColumnStat[]) : []),
    [preview?.column_stats],
  );

  const columnStatsMap = useMemo(() => {
    const map = new Map<string, PipelinePreviewColumnStat>();
    columnStats.forEach((stat) => {
      if (stat?.name) {
        map.set(stat.name, stat);
      }
    });
    return map;
  }, [columnStats]);

  const resolvedFeatures = useMemo(() => {
    if (activeSignal?.feature_columns?.length) {
      return activeSignal.feature_columns;
    }
    if (configuredFeatures.length) {
      return configuredFeatures;
    }
    if (resolvedTarget) {
      return previewColumns.filter((column) => column !== resolvedTarget);
    }
    return previewColumns.slice();
  }, [activeSignal?.feature_columns, configuredFeatures, previewColumns, resolvedTarget]);

  const autoIncludedFeatures = activeSignal?.auto_included_columns ?? [];
  const missingConfiguredFeatures = activeSignal?.missing_feature_columns ?? [];
  const excludedColumns = activeSignal?.excluded_columns ?? [];
  const targetMissingCount = activeSignal?.target_missing_count ?? 0;
  const targetMissingPercentage =
    typeof activeSignal?.target_missing_percentage === 'number'
      ? activeSignal.target_missing_percentage
      : null;
  const targetDtype = activeSignal?.target_dtype ?? (resolvedTarget ? columnStatsMap.get(resolvedTarget)?.dtype ?? null : null);

  const previewRowCount = typeof preview?.metrics?.row_count === 'number' ? preview.metrics.row_count : null;
  const featureCount = resolvedFeatures.length;
  const autoCount = autoIncludedFeatures.length;
  const excludedCount = excludedColumns.length;
  const missingConfiguredCount = missingConfiguredFeatures.length;

  const splitSummary = useMemo(() => findSplitSummary(appliedSteps), [appliedSteps]);

  const summaryItems = useMemo(() => {
    const items: { label: string; value: string }[] = [];
    items.push({ label: 'Target column', value: resolvedTarget || 'Not set' });
    items.push({ label: 'Feature columns', value: formatMetricValue(featureCount) });
    if (previewRowCount !== null) {
      items.push({ label: 'Preview rows', value: formatMetricValue(previewRowCount) });
    }
    if (autoCount) {
      items.push({ label: 'Auto-selected features', value: formatMetricValue(autoCount) });
    }
    if (missingConfiguredCount) {
      items.push({ label: 'Configured features missing', value: formatMetricValue(missingConfiguredCount) });
    }
    if (excludedCount) {
      items.push({ label: 'Excluded columns', value: formatMetricValue(excludedCount) });
    }
    if (targetMissingCount) {
      const formattedMissing = formatMetricValue(targetMissingCount);
      const formattedPercentage = targetMissingPercentage !== null ? formatMissingPercentage(targetMissingPercentage) : null;
      items.push({
        label: 'Target missing values',
        value: formattedPercentage ? `${formattedMissing} (${formattedPercentage})` : formattedMissing,
      });
    }
    return items;
  }, [
    autoCount,
    excludedCount,
    featureCount,
    formatMetricValue,
    formatMissingPercentage,
    missingConfiguredCount,
    previewRowCount,
    resolvedTarget,
    targetMissingCount,
    targetMissingPercentage,
  ]);

  const connectionNote = useMemo(() => {
    if (!sourceId) {
      return 'Connect this node to a dataset to confirm feature and target mappings.';
    }
    if (!hasReachableSource) {
      return 'Link this node to the upstream dataset to resolve feature columns.';
    }
    return null;
  }, [hasReachableSource, sourceId]);

  const previewIdleNote = useMemo(() => {
    if (!sourceId || !hasReachableSource) {
      return null;
    }
    if (previewStatus === 'idle') {
      return 'Refresh the preview to surface feature/target diagnostics.';
    }
    return null;
  }, [hasReachableSource, previewStatus, sourceId]);

  const targetMissingNote = useMemo(() => {
    if (!resolvedTarget) {
      return 'Set a target column so downstream nodes understand the supervised objective.';
    }
    if (previewStatus === 'success' && resolvedTarget && !previewColumns.includes(resolvedTarget)) {
      return `Target column "${resolvedTarget}" is not present in the current preview output.`;
    }
    return null;
  }, [previewColumns, previewStatus, resolvedTarget]);

  const canRefreshPreview = Boolean(sourceId) && hasReachableSource && previewStatus !== 'loading';

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Feature/target separation</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onRefreshPreview}
            disabled={!canRefreshPreview}
          >
            {previewStatus === 'loading' ? 'Refreshing...' : 'Refresh preview'}
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Confirm which columns feed the feature matrix (X) and which column serves as the supervised target (y).
        This metadata helps downstream modeling nodes stay aligned with the configured training objective.
      </p>
      {connectionNote && <p className="canvas-modal__note canvas-modal__note--warning">{connectionNote}</p>}
      {previewState.error && previewStatus === 'error' && (
        <p className="canvas-modal__note canvas-modal__note--error">{previewState.error}</p>
      )}
      {previewIdleNote && <p className="canvas-modal__note canvas-modal__note--info">{previewIdleNote}</p>}
      {targetMissingNote && <p className="canvas-modal__note canvas-modal__note--warning">{targetMissingNote}</p>}
      {missingConfiguredCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Missing configured feature columns: {summarizeList(missingConfiguredFeatures)}.
        </p>
      )}
      {excludedCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Columns excluded from the feature matrix: {summarizeList(excludedColumns)}.
        </p>
      )}
      {autoCount > 0 && configuredFeatures.length === 0 && (
        <p className="canvas-modal__note canvas-modal__note--info">
          Auto-selected feature columns: {summarizeList(autoIncludedFeatures)}.
        </p>
      )}
      {splitSummary && <p className="canvas-modal__note canvas-modal__note--info">{splitSummary}</p>}

      {summaryItems.length > 0 && (
        <ul className="canvas-modal__note-list">
          {summaryItems.map((item) => (
            <li key={`feature-target-summary-${item.label}`}>
              {item.label}: <strong>{item.value}</strong>
            </li>
          ))}
        </ul>
      )}

      <div className="canvas-modal__parameter-grid">
        {targetColumnParameter && renderParameterField(targetColumnParameter)}
      </div>
      {previewStatus === 'success' && featureCount === 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          No feature columns detected. Confirm that the dataset includes columns other than the target.
        </p>
      )}

      {resolvedTarget && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Target dtype: {targetDtype ? <strong>{targetDtype}</strong> : <span>Unknown</span>}
          {targetMissingCount
            ? ` - Missing rows flagged for review: ${formatMetricValue(targetMissingCount)}.`
            : ''}
        </p>
      )}
    </section>
  );
};
