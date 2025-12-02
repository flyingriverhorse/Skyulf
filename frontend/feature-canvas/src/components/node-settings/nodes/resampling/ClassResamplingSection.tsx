import React, { useMemo, useEffect } from 'react';
import type { FeatureNodeParameter, FeatureNodeParameterOption } from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import {
  extractPendingConfigurationDetails,
  type PendingConfigurationDetail,
} from '../../utils/pendingConfiguration';

type ResamplingMode = 'undersampling' | 'oversampling';

type ResamplingSchemaGuardDetail = {
  name: string;
  logical_family: string;
};

export type ResamplingSchemaGuard = {
  blocked: boolean;
  message: string;
  columns: string[];
  details: ResamplingSchemaGuardDetail[];
};

export type ClassResamplingConfig = {
  method: string;
  targetColumn: string;
  samplingStrategy: number | string | null;
  randomState: number | null;
  replacement: boolean | null;
  kNeighbors: number | null;
} | null;

type ClassResamplingSectionProps = {
  mode?: ResamplingMode;
  sourceId?: string | null;
  hasReachableSource: boolean;
  previewState: PreviewState;
  onRefreshPreview: () => void;
  config: ClassResamplingConfig;
  methodParameter: FeatureNodeParameter | null;
  targetColumnParameter: FeatureNodeParameter | null;
  samplingStrategyParameter: FeatureNodeParameter | null;
  randomStateParameter: FeatureNodeParameter | null;
  replacementParameter: FeatureNodeParameter | null;
  kNeighborsParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  formatMissingPercentage: (value?: number | null) => string;
  schemaGuard?: ResamplingSchemaGuard | null;
  onPendingConfigurationWarning?: (
    details: PendingConfigurationDetail[]
  ) => void;
  onPendingConfigurationCleared?: () => void;
};

type ClassCountRow = {
  label: string;
  count: number;
  percentage: number;
  isMissing: boolean;
};

const STRATEGY_DESCRIPTIONS: Record<ResamplingMode, Record<string, string>> = {
  undersampling: {
    auto: 'auto (match majority to the smallest class)',
    majority: 'majority (down-sample only the majority class)',
    'not minority': 'not minority (down-sample all but the minority class)',
    'not majority': 'not majority (down-sample every class except the majority)',
    all: 'all (down-sample every class)',
  },
  oversampling: {
    auto: 'auto (boost minority class to majority size)',
    minority: 'minority (only over-sample the minority class)',
    'not minority': 'not minority (over-sample every class except the minority)',
    'not majority': 'not majority (avoid synthesising the majority class)',
    all: 'all (over-sample all classes equally)',
  },
};

const normalizeStrategyLabel = (
  value: number | string | null | undefined,
  mode: ResamplingMode,
): string => {
  if (value === null || value === undefined || value === '') {
    return STRATEGY_DESCRIPTIONS[mode].auto;
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return STRATEGY_DESCRIPTIONS[mode].auto;
    }
    const clamped = Math.max(Math.min(value, 1), 0);
    const percentage = clamped * 100;
    const formattedRatio = clamped.toFixed(clamped >= 0.1 || clamped === 0 ? 2 : 3);
    const descriptor = mode === 'undersampling' ? 'majority' : 'majority baseline';
    return `${formattedRatio} ratio (${percentage.toFixed(1)}% of ${descriptor})`;
  }

  const normalized = value.trim().toLowerCase();
  return STRATEGY_DESCRIPTIONS[mode][normalized] ?? normalized;
};

const findResamplingSummary = (appliedSteps: unknown[], mode: ResamplingMode): string | null => {
  const keywords = mode === 'oversampling'
    ? ['over-sampling', 'oversampling', 'smote', 'adasyn', 'tomek', 'svm smote', 'kmeans smote', 'borderline smote']
    : ['under-sampling', 'undersampling', 'resampling'];
  for (let index = appliedSteps.length - 1; index >= 0; index -= 1) {
    const entry = appliedSteps[index];
    if (typeof entry !== 'string') {
      continue;
    }
    const lowered = entry.toLowerCase();
    if (keywords.some((keyword) => lowered.includes(keyword))) {
      return entry;
    }
  }
  return null;
};

const describeClassLabel = (label: string, isMissing: boolean): string => {
  if (isMissing) {
    return 'Missing';
  }
  if (!label.length) {
    return 'Empty string';
  }
  return label;
};

export const ClassResamplingSection: React.FC<ClassResamplingSectionProps> = ({
  mode = 'undersampling',
  sourceId,
  hasReachableSource,
  previewState,
  onRefreshPreview,
  config,
  methodParameter,
  targetColumnParameter,
  samplingStrategyParameter,
  randomStateParameter,
  replacementParameter,
  kNeighborsParameter,
  renderParameterField,
  formatMetricValue,
  formatMissingPercentage,
  schemaGuard,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (!previewState.data?.signals?.full_execution) {
      onPendingConfigurationCleared?.();
      return;
    }

    const details = extractPendingConfigurationDetails(
      previewState.data.signals.full_execution,
    );

    if (details.length > 0) {
      onPendingConfigurationWarning?.(details);
    } else {
      onPendingConfigurationCleared?.();
    }
  }, [
    previewState.data?.signals?.full_execution,
    onPendingConfigurationWarning,
    onPendingConfigurationCleared,
  ]);
  const preview = previewState.data;
  const previewStatus = previewState.status;
  const appliedSteps = Array.isArray(preview?.applied_steps) ? preview?.applied_steps : [];
  const resamplingSummary = useMemo(
    () => findResamplingSummary(appliedSteps, mode),
    [appliedSteps, mode],
  );

  const targetColumn = config?.targetColumn ?? '';
  const methodKey = config?.method || (typeof methodParameter?.default === 'string' ? methodParameter.default : '');
  const methodLabel = useMemo(() => {
    const options: FeatureNodeParameterOption[] = Array.isArray(methodParameter?.options)
      ? (methodParameter?.options as FeatureNodeParameterOption[])
      : [];
    if (methodKey) {
      const match = options.find((option: FeatureNodeParameterOption) => option.value === methodKey);
      if (match?.label) {
        return match.label;
      }
      return methodKey.replace(/_/g, ' ');
    }
    if (options.length > 0) {
      return options[0].label ?? options[0].value;
    }
    return mode === 'undersampling' ? 'Random under-sampling' : 'Synthetic over-sampling';
  }, [methodKey, methodParameter?.options, mode]);

  const strategyLabel = useMemo(
    () => normalizeStrategyLabel(config?.samplingStrategy, mode),
    [config?.samplingStrategy, mode],
  );

  const randomStateLabel = useMemo(() => {
    if (config?.randomState === null || config?.randomState === undefined) {
      return 'Not set';
    }
    if (Number.isFinite(config.randomState)) {
      return String(config.randomState);
    }
    return 'Not set';
  }, [config?.randomState]);

  const kNeighborsLabel = useMemo(() => {
    if (mode !== 'oversampling') {
      return null;
    }
    const configured = config?.kNeighbors;
    if (typeof configured === 'number' && Number.isFinite(configured)) {
      return String(Math.max(1, Math.round(configured)));
    }
    if (kNeighborsParameter && typeof kNeighborsParameter.default === 'number') {
      const fallback = kNeighborsParameter.default;
      if (Number.isFinite(fallback)) {
        return String(Math.max(1, Math.round(fallback)));
      }
    }
    return 'Default';
  }, [config?.kNeighbors, kNeighborsParameter, mode]);

  const replacementLabel =
    config?.replacement === null || config?.replacement === undefined
      ? null
      : config.replacement
        ? 'With replacement'
        : 'Without replacement';

  const classCountSummary = useMemo(() => {
    if (previewStatus !== 'success' || !preview || !targetColumn) {
      return { rows: [] as ClassCountRow[], total: 0, truncated: false };
    }

    const sampleRows = Array.isArray(preview.sample_rows) ? preview.sample_rows : [];
    if (!sampleRows.length) {
      return { rows: [] as ClassCountRow[], total: 0, truncated: false };
    }

    const counts = new Map<string, ClassCountRow>();

    sampleRows.forEach((row: Record<string, unknown>) => {
      if (!row || typeof row !== 'object') {
        return;
      }
      const rawValue = (row as Record<string, unknown>)[targetColumn];
      const isMissing = rawValue === null || rawValue === undefined;
      const key = isMissing ? '__missing__' : String(rawValue);
      const label = describeClassLabel(isMissing ? '' : key, isMissing);
      const existing = counts.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        counts.set(
          key,
          {
            label,
            count: 1,
            percentage: 0,
            isMissing,
          },
        );
      }
    });

    const aggregate = Array.from(counts.values()).sort((a, b) => b.count - a.count);
    const totalCount = aggregate.reduce((acc, entry) => acc + entry.count, 0);
    const limit = 8;
    const truncated = aggregate.length > limit;
    const rows = truncated ? aggregate.slice(0, limit) : aggregate;
    const rowsWithPercentage = rows.map((entry) => ({
      ...entry,
      percentage: totalCount > 0 ? (entry.count / totalCount) * 100 : 0,
    }));

    return {
      rows: rowsWithPercentage,
      total: totalCount,
      truncated,
    };
  }, [preview, previewStatus, targetColumn]);

  const previewRowCount = preview?.metrics?.row_count ?? null;
  const canRefreshPreview = Boolean(sourceId) && hasReachableSource && previewStatus !== 'loading' && !schemaGuard?.blocked;

  const connectionNote = useMemo(() => {
    if (!sourceId) {
      return 'Connect to a dataset to preview class distributions.';
    }
    if (!hasReachableSource) {
      return 'Link this node to the dataset input to evaluate class balance.';
    }
    return null;
  }, [hasReachableSource, sourceId]);

  const schemaGuardSummary = useMemo(() => {
    if (!schemaGuard?.blocked || !schemaGuard.details.length) {
      return '';
    }
    return schemaGuard.details
      .map((detail: ResamplingSchemaGuardDetail) =>
        detail.logical_family ? `${detail.name} (${detail.logical_family})` : detail.name,
      )
      .join(', ');
  }, [schemaGuard]);

  const targetNotConfigured = !targetColumn;
  const previewMissingTarget =
    Boolean(targetColumn) && Array.isArray(preview?.columns) && !preview?.columns.includes(targetColumn);

  const heading = mode === 'undersampling' ? 'Class undersampling' : 'Class oversampling';
  const description =
    mode === 'undersampling'
      ? 'Random under-sampling trims majority-class examples to mitigate imbalance before training downstream models. It preserves minority class signals but can discard information from majority classes when ratios are aggressive.'
      : 'Synthetic over-sampling boosts minority representation by generating new samples. Integer feature columns are temporarily cast to float so synthesized values can include decimals, which helps models learn rare classes while requiring careful preprocessing to avoid noise.';
  const connectionGuidance = mode === 'undersampling'
    ? 'enable undersampling'
    : 'enable oversampling';

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>{heading}</h3>
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
      <p className="canvas-modal__note">{description}</p>
      {connectionNote && <p className="canvas-modal__note canvas-modal__note--warning">{connectionNote}</p>}
      {targetNotConfigured && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Specify a target column to compute class ratios and {connectionGuidance}.
        </p>
      )}
      {schemaGuard?.blocked && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {schemaGuard.message}
          {schemaGuardSummary && <>
            {' '}Columns: {schemaGuardSummary}.
          </>}
        </p>
      )}
      {previewStatus === 'error' && previewState.error && (
        <p className="canvas-modal__note canvas-modal__note--error">{previewState.error}</p>
      )}
      {previewMissingTarget && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Target column “{targetColumn}” is not present in the current preview. Check upstream nodes or refresh the
          dataset snapshot.
        </p>
      )}
      {resamplingSummary && (
        <p className="canvas-modal__note canvas-modal__note--info">{resamplingSummary}</p>
      )}

      <ul className="canvas-modal__note-list">
        <li>
          Method: <strong>{methodLabel}</strong>
        </li>
        <li>
          Sampling strategy: <strong>{strategyLabel}</strong>
        </li>
        {mode === 'oversampling' && kNeighborsLabel && (
          <li>
            k-neighbors: <strong>{kNeighborsLabel}</strong>
          </li>
        )}
        {replacementLabel && (
          <li>
            Replacement: <strong>{replacementLabel}</strong>
          </li>
        )}
        <li>
          Random state: <strong>{randomStateLabel}</strong>
        </li>
      </ul>

      <div className="canvas-modal__parameter-grid">
        {methodParameter && renderParameterField(methodParameter)}
        {samplingStrategyParameter && renderParameterField(samplingStrategyParameter)}
        {mode === 'oversampling' && kNeighborsParameter && renderParameterField(kNeighborsParameter)}
        {randomStateParameter && renderParameterField(randomStateParameter)}
        {replacementParameter && renderParameterField(replacementParameter)}
      </div>

      {targetColumnParameter && (
        <div className="canvas-modal__parameter-list">
          {renderParameterField(targetColumnParameter)}
        </div>
      )}

      {targetColumn && previewStatus === 'success' && !previewMissingTarget && (
        classCountSummary.rows.length ? (
          <div className="canvas-cast__table-wrapper">
            <table className="canvas-cast__table">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col">Sample count</th>
                  <th scope="col">Share</th>
                </tr>
              </thead>
              <tbody>
                {classCountSummary.rows.map((entry) => (
                  <tr key={`resampling-class-${entry.label}`}>
                    <th scope="row">{entry.label}</th>
                    <td>{formatMetricValue(entry.count)}</td>
                    <td>{formatMissingPercentage(entry.percentage)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="canvas-modal__note">
              Analysed {formatMetricValue(classCountSummary.total)} row
              {classCountSummary.total === 1 ? '' : 's'} from the preview.
              {classCountSummary.truncated ? ' Showing top classes by frequency.' : ''}
            </p>
            {previewRowCount !== null && (
              <p className="canvas-modal__note canvas-modal__note--muted">
                Preview rows after resampling: {formatMetricValue(previewRowCount)}.
              </p>
            )}
          </div>
        ) : (
          <p className="canvas-modal__note">
            No class distribution available in the preview sample. Refresh after configuring the node or increasing
            the sample size.
          </p>
        )
      )}
    </section>
  );
};