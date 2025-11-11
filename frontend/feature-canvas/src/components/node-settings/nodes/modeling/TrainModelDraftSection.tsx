import React, { useMemo } from 'react';
import type {
  FeatureNodeParameter,
  PipelinePreviewColumnSchema,
  PipelinePreviewColumnStat,
} from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';

type ProblemType = 'classification' | 'regression';

export type TrainModelDraftConfig = {
  targetColumn: string;
  problemType: ProblemType;
};

type TrainModelDraftSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  previewState: PreviewState;
  onRefreshPreview: () => void;
  config: TrainModelDraftConfig | null;
  targetColumnParameter: FeatureNodeParameter | null;
  problemTypeParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  formatMissingPercentage: (value?: number | null) => string;
  schemaColumns: PipelinePreviewColumnSchema[];
};

const PROBLEM_TYPE_LABEL: Record<ProblemType, string> = {
  classification: 'Classification',
  regression: 'Regression',
};

const isNumericFamily = (family: PipelinePreviewColumnSchema['logical_family'] | undefined): boolean => {
  return family === 'numeric' || family === 'integer';
};

const isNumericDtype = (dtype: string | null | undefined): boolean => {
  if (!dtype) {
    return false;
  }
  const lowered = dtype.toLowerCase();
  return (
    lowered.includes('int') ||
    lowered.includes('float') ||
    lowered.includes('double') ||
    lowered.includes('decimal') ||
    lowered.includes('number')
  );
};

const summarizeList = (values: string[], limit = 3): string => {
  if (values.length === 0) {
    return '';
  }
  if (values.length <= limit) {
    return values.join(', ');
  }
  const preview = values.slice(0, limit).join(', ');
  return `${preview}, ...`;
};

const findTrainSummary = (steps: unknown[]): string | null => {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const entry = steps[index];
    if (typeof entry !== 'string') {
      continue;
    }
    if (entry.toLowerCase().startsWith('train model draft')) {
      return entry;
    }
  }
  return null;
};

export const TrainModelDraftSection: React.FC<TrainModelDraftSectionProps> = ({
  sourceId,
  hasReachableSource,
  previewState,
  onRefreshPreview,
  config,
  targetColumnParameter,
  problemTypeParameter,
  renderParameterField,
  formatMetricValue,
  formatMissingPercentage,
  schemaColumns,
}) => {
  const preview = previewState.data;
  const previewStatus = previewState.status;
  const appliedSteps = Array.isArray(preview?.applied_steps) ? preview.applied_steps : [];
  const columnStats: PipelinePreviewColumnStat[] = Array.isArray(preview?.column_stats)
    ? preview.column_stats
    : [];
  const metrics = preview?.metrics ?? null;

  const previewColumnNames = useMemo(() => {
    if (!Array.isArray(preview?.columns)) {
      return [] as string[];
    }
    return preview.columns.filter((name): name is string => typeof name === 'string' && name.trim().length > 0);
  }, [preview?.columns]);

  const effectiveSchemaColumns = useMemo(() => {
    if (schemaColumns.length) {
      return schemaColumns;
    }
    if (Array.isArray(preview?.schema?.columns)) {
      return preview.schema.columns as PipelinePreviewColumnSchema[];
    }
    if (previewColumnNames.length > 0) {
      // Fallback to raw column names when the backend omits the schema payload for lightweight previews.
      return previewColumnNames.map((name) => ({
        name,
        pandas_dtype: null,
        logical_family: 'unknown' as const,
        nullable: true,
      }));
    }
    return [] as PipelinePreviewColumnSchema[];
  }, [preview?.schema?.columns, previewColumnNames, schemaColumns]);

  const schemaByName = useMemo(() => {
    const map = new Map<string, PipelinePreviewColumnSchema>();
    effectiveSchemaColumns.forEach((column) => {
      if (column?.name) {
        map.set(column.name, column);
      }
    });
    return map;
  }, [effectiveSchemaColumns]);

  const targetColumn = config?.targetColumn ?? '';
  const selectedProblemType: ProblemType = config?.problemType ?? 'classification';

  const targetStat = targetColumn
    ? columnStats.find((stat) => stat?.name === targetColumn) ?? null
    : null;
  const targetSchema = targetColumn ? schemaByName.get(targetColumn) ?? null : null;
  const targetPresentInSchema = Boolean(targetSchema);

  const featureNames = useMemo(() => {
    const names = new Set<string>();
    columnStats.forEach((stat) => {
      if (stat?.name && stat.name !== targetColumn) {
        names.add(stat.name);
      }
    });
    effectiveSchemaColumns.forEach((column) => {
      if (column?.name && column.name !== targetColumn) {
        names.add(column.name);
      }
    });
    return Array.from(names).sort();
  }, [columnStats, effectiveSchemaColumns, targetColumn]);

  const numericFeatureNames = useMemo(() => {
    return featureNames.filter((name) => {
      const schema = schemaByName.get(name);
      if (schema && isNumericFamily(schema.logical_family)) {
        return true;
      }
      const stat = columnStats.find((entry) => entry.name === name);
      return isNumericDtype(stat?.dtype ?? null);
    });
  }, [columnStats, featureNames, schemaByName]);

  const numericFeatureSet = useMemo(() => new Set(numericFeatureNames), [numericFeatureNames]);

  const nonNumericFeatureNames = useMemo(
    () => featureNames.filter((name) => !numericFeatureSet.has(name)),
    [featureNames, numericFeatureSet],
  );

  const featureMissingNames = useMemo(
    () =>
      columnStats
        .filter((stat) => stat?.name && stat.name !== targetColumn && (stat.missing_count ?? 0) > 0)
        .map((stat) => stat.name),
    [columnStats, targetColumn],
  );

  const previewRowCount = typeof metrics?.row_count === 'number' ? metrics.row_count : null;
  const targetDistinctCount =
    typeof targetStat?.distinct_count === 'number' && Number.isFinite(targetStat.distinct_count)
      ? targetStat.distinct_count
      : null;
  const missingTargetCount = typeof targetStat?.missing_count === 'number' ? targetStat.missing_count : 0;
  const targetMissingPercentage =
    typeof targetStat?.missing_percentage === 'number'
      ? targetStat.missing_percentage
      : null;

  const readinessIssues = useMemo(() => {
    const issues: string[] = [];

    if (!targetColumn) {
      issues.push('Set a target column to evaluate downstream modeling constraints.');
      return issues;
    }

    if (previewStatus !== 'success') {
      return issues;
    }

    // If target column is not in preview but we have features, it might have been split upstream
    // Only warn if we have no features at all
    if (!targetStat && !targetPresentInSchema && !featureNames.length) {
      issues.push(`Target column "${targetColumn}" is not present in the preview output, and no feature columns were found.`);
      return issues;
    }

    if (!featureNames.length) {
      issues.push('Add at least one feature column in addition to the target before training.');
    }

    if (targetStat && missingTargetCount > 0) {
      issues.push(`Target column includes ${missingTargetCount} missing value${missingTargetCount === 1 ? '' : 's'}.`);
    }

    if (selectedProblemType === 'classification' && targetStat && (targetDistinctCount ?? 0) < 2) {
      issues.push('Classification tasks require at least two distinct target classes.');
    }

    if (selectedProblemType === 'regression' && numericFeatureNames.length === 0) {
      issues.push('Regression tasks need at least one numeric feature column.');
    }

    if (featureMissingNames.length > 0) {
      issues.push(`Feature columns with missing values: ${summarizeList(featureMissingNames)}.`);
    }

    if (previewRowCount !== null && previewRowCount === 0) {
      issues.push('Preview returned zero rows after upstream transformations.');
    }

    return issues;
  }, [
    featureMissingNames,
    featureNames.length,
    missingTargetCount,
    numericFeatureNames.length,
    previewRowCount,
    previewStatus,
    selectedProblemType,
    targetColumn,
    targetDistinctCount,
    targetStat,
    targetPresentInSchema,
  ]);

  const detectionLabel = useMemo(() => PROBLEM_TYPE_LABEL[selectedProblemType], [selectedProblemType]);

  const summaryItems = useMemo(() => {
    const items: { label: string; value: string }[] = [
      { label: 'Problem type', value: detectionLabel },
      { label: 'Target column', value: targetColumn || 'Not set' },
    ];

    if (previewStatus === 'success') {
      if (targetDistinctCount !== null) {
        items.push({ label: 'Target distinct values', value: formatMetricValue(targetDistinctCount) });
      }
      if (targetStat) {
        const missingCountLabel = formatMetricValue(missingTargetCount);
        const missingPercentageLabel =
          targetMissingPercentage !== null ? formatMissingPercentage(targetMissingPercentage) : null;
        items.push({
          label: 'Target missing values',
          value: missingPercentageLabel ? `${missingCountLabel} (${missingPercentageLabel})` : missingCountLabel,
        });
      }
      if (featureMissingNames.length > 0) {
        items.push({
          label: 'Features with missing values',
          value: formatMetricValue(featureMissingNames.length),
        });
      }
    }

    return items;
  }, [
    detectionLabel,
    featureMissingNames.length,
    formatMetricValue,
    formatMissingPercentage,
    missingTargetCount,
    previewStatus,
    targetColumn,
    targetDistinctCount,
    targetMissingPercentage,
    targetStat,
  ]);

  const connectionNote = useMemo(() => {
    if (!sourceId) {
      return 'Connect this node to a dataset to validate training readiness.';
    }
    if (!hasReachableSource) {
      return 'Link this node to upstream transforms to analyse the feature set that reaches it.';
    }
    return null;
  }, [hasReachableSource, sourceId]);

  const previewIdleNote = useMemo(() => {
    if (!sourceId || !hasReachableSource) {
      return null;
    }
    if (previewStatus === 'idle') {
      return 'Refresh the preview to surface downstream modeling checks.';
    }
    return null;
  }, [hasReachableSource, previewStatus, sourceId]);

  const trainSummary = useMemo(() => findTrainSummary(appliedSteps), [appliedSteps]);
  const canRefreshPreview = Boolean(sourceId) && hasReachableSource && previewStatus !== 'loading';
  const readyMessage = useMemo(() => {
    if (!targetColumn || previewStatus !== 'success' || readinessIssues.length > 0) {
      return null;
    }
    return 'No critical blockers detected in the preview. Proceed with downstream training when ready.';
  }, [previewStatus, readinessIssues.length, targetColumn]);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Train model readiness</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onRefreshPreview}
            disabled={!canRefreshPreview}
          >
            {previewStatus === 'loading' ? 'Refreshing…' : 'Refresh preview'}
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Validate that the pipeline output is ready for lightweight training by checking the target column,
        feature balance, and missing values against the cached preview.
      </p>
      {connectionNote && <p className="canvas-modal__note canvas-modal__note--warning">{connectionNote}</p>}
      {previewState.error && previewStatus === 'error' && (
        <p className="canvas-modal__note canvas-modal__note--error">{previewState.error}</p>
      )}
      {previewIdleNote && <p className="canvas-modal__note canvas-modal__note--info">{previewIdleNote}</p>}
      {readinessIssues.map((issue, index) => (
        <p key={`train-model-issue-${index}`} className="canvas-modal__note canvas-modal__note--warning">
          {issue}
        </p>
      ))}
      {readyMessage && <p className="canvas-modal__note canvas-modal__note--info">{readyMessage}</p>}
      {trainSummary && (
        <p className="canvas-modal__note canvas-modal__note--muted">{trainSummary}</p>
      )}

      <div className="canvas-modal__parameter-grid">
        {targetColumnParameter && renderParameterField(targetColumnParameter)}
        {problemTypeParameter && renderParameterField(problemTypeParameter)}
      </div>

      {summaryItems.length > 0 && (
        <ul className="canvas-modal__note-list">
          {summaryItems.map((item) => (
            <li key={`train-model-summary-${item.label}`}>
              {item.label}: <strong>{item.value}</strong>
            </li>
          ))}
        </ul>
      )}

      {previewStatus === 'success' && featureMissingNames.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Features with missing values: {summarizeList(featureMissingNames)}.
        </p>
      )}

      {previewStatus === 'success' && selectedProblemType === 'regression' && nonNumericFeatureNames.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Non-numeric features detected: {summarizeList(nonNumericFeatureNames)}. Consider encoding or casting them
          before training regression models.
        </p>
      )}
    </section>
  );
};
