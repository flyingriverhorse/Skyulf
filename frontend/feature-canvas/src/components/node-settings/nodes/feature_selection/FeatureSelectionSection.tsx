import React, { useMemo } from 'react';
import type { FeatureNodeParameter, FeatureSelectionNodeSignal } from '../../../../api';

type FeatureSelectionSectionProps = {
  columnsParameter: FeatureNodeParameter | null;
  autoDetectParameter: FeatureNodeParameter | null;
  targetColumnParameter: FeatureNodeParameter | null;
  methodParameter: FeatureNodeParameter | null;
  scoreFuncParameter: FeatureNodeParameter | null;
  problemTypeParameter: FeatureNodeParameter | null;
  kParameter: FeatureNodeParameter | null;
  percentileParameter: FeatureNodeParameter | null;
  alphaParameter: FeatureNodeParameter | null;
  thresholdParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  estimatorParameter: FeatureNodeParameter | null;
  stepParameter: FeatureNodeParameter | null;
  minFeaturesParameter: FeatureNodeParameter | null;
  maxFeaturesParameter: FeatureNodeParameter | null;
  dropUnselectedParameter: FeatureNodeParameter | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  signal: FeatureSelectionNodeSignal | null;
  canResetNode?: boolean;
  onResetNode?: () => void;
};

const isDefinedParameter = (
  parameter: FeatureNodeParameter | null,
): parameter is FeatureNodeParameter => Boolean(parameter);

const formatNumber = (value: number | null | undefined, fractionDigits = 4): string => {
  if (value === null || value === undefined) {
    return '—';
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: fractionDigits });
};

const formatPValue = (value: number | null | undefined): string => {
  if (value === null || value === undefined) {
    return '—';
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  if (numeric === 0) {
    return '< 1e-12';
  }
  if (numeric < 0.0001) {
    return numeric.toExponential(2);
  }
  return numeric.toLocaleString(undefined, { maximumFractionDigits: 6 });
};

const formatColumnPreview = (columns: string[], limit = 8): string => {
  if (!columns.length) {
    return 'None';
  }
  const preview = columns.slice(0, limit);
  const remainder = columns.length - preview.length;
  return remainder > 0 ? `${preview.join(', ')} … (${remainder} more)` : preview.join(', ');
};

export const FeatureSelectionSection: React.FC<FeatureSelectionSectionProps> = ({
  columnsParameter,
  autoDetectParameter,
  targetColumnParameter,
  methodParameter,
  scoreFuncParameter,
  problemTypeParameter,
  kParameter,
  percentileParameter,
  alphaParameter,
  thresholdParameter,
  modeParameter,
  estimatorParameter,
  stepParameter,
  minFeaturesParameter,
  maxFeaturesParameter,
  dropUnselectedParameter,
  renderParameterField,
  signal,
  canResetNode = false,
  onResetNode,
}) => {
  const primaryParameters = useMemo(() => {
    return [
      columnsParameter,
      autoDetectParameter,
      targetColumnParameter,
      methodParameter,
      scoreFuncParameter,
      problemTypeParameter,
    ].filter(isDefinedParameter);
  }, [
    columnsParameter,
    autoDetectParameter,
    targetColumnParameter,
    methodParameter,
    scoreFuncParameter,
    problemTypeParameter,
  ]);

  const tuningParameters = useMemo(() => {
    return [
      kParameter,
      percentileParameter,
      alphaParameter,
      thresholdParameter,
      modeParameter,
      estimatorParameter,
      stepParameter,
      minFeaturesParameter,
      maxFeaturesParameter,
      dropUnselectedParameter,
    ].filter(isDefinedParameter);
  }, [
    kParameter,
    percentileParameter,
    alphaParameter,
    thresholdParameter,
    modeParameter,
    estimatorParameter,
    stepParameter,
    minFeaturesParameter,
    maxFeaturesParameter,
    dropUnselectedParameter,
  ]);

  const configuredColumns = signal?.configured_columns ?? [];
  const evaluatedColumns = signal?.evaluated_columns ?? [];
  const selectedColumns = signal?.selected_columns ?? [];
  const droppedColumns = signal?.dropped_columns ?? [];
  const summaries = signal?.feature_summaries ?? [];
  const notes = signal?.notes ?? [];
  const dropUnselected = signal?.drop_unselected ?? false;

  const evaluationSummary = useMemo(() => {
    if (!evaluatedColumns.length && !configuredColumns.length) {
      return null;
    }
    if (evaluatedColumns.length === configuredColumns.length) {
      return `Evaluated ${evaluatedColumns.length.toLocaleString()} column(s).`;
    }
    if (!configuredColumns.length) {
      return `Evaluated ${evaluatedColumns.length.toLocaleString()} column(s) via auto-detect.`;
    }
    const additional = Math.max(evaluatedColumns.length - configuredColumns.length, 0);
    if (additional > 0) {
      return `Evaluated ${evaluatedColumns.length.toLocaleString()} column(s) (${additional.toLocaleString()} auto-detected).`;
    }
    return `Evaluated ${evaluatedColumns.length.toLocaleString()} of ${configuredColumns.length.toLocaleString()} configured column(s).`;
  }, [configuredColumns, evaluatedColumns]);

  const methodSummary = useMemo(() => {
    if (!signal) {
      return null;
    }
    const parts: string[] = [];
    if (signal.method) {
      parts.push(`Method: ${signal.method}`);
    }
    if (signal.score_func) {
      parts.push(`Score: ${signal.score_func}`);
    }
    if (signal.estimator) {
      parts.push(`Estimator: ${signal.estimator}`);
    }
    if (signal.problem_type && signal.problem_type !== 'auto') {
      parts.push(`Problem: ${signal.problem_type}`);
    }
    if (signal.target_column) {
      parts.push(`Target: ${signal.target_column}`);
    }
    if (typeof signal.k === 'number' && Number.isFinite(signal.k)) {
      parts.push(`k = ${signal.k.toLocaleString()}`);
    }
    if (typeof signal.percentile === 'number' && Number.isFinite(signal.percentile)) {
      parts.push(`percentile = ${formatNumber(signal.percentile, 2)}`);
    }
    if (typeof signal.alpha === 'number' && Number.isFinite(signal.alpha)) {
      parts.push(`alpha = ${formatNumber(signal.alpha, 4)}`);
    }
    if (signal.threshold !== null && signal.threshold !== undefined && Number.isFinite(signal.threshold)) {
      parts.push(`threshold = ${formatNumber(signal.threshold, 4)}`);
    }
    if (signal.auto_detect) {
      parts.push('Auto-detect enabled');
    }
    return parts.length ? parts.join(' • ') : null;
  }, [signal]);

  const selectedSummary = useMemo(() => formatColumnPreview(selectedColumns), [selectedColumns]);
  const droppedSummary = useMemo(() => formatColumnPreview(droppedColumns), [droppedColumns]);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Feature selection settings</h3>
        {canResetNode && (
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onResetNode}
            disabled={!onResetNode}
          >
            Reset node
          </button>
        )}
      </div>
      <p className="canvas-modal__note">
        Score candidate predictors and retain the most informative columns using statistical tests or model-based selectors.
        Run a preview after adjusting parameters to refresh rankings.
      </p>
      {methodSummary && (
        <p className="canvas-modal__note canvas-modal__note--info">{methodSummary}</p>
      )}
      {evaluationSummary && (
        <p className="canvas-modal__note canvas-modal__note--info">{evaluationSummary}</p>
      )}
      {primaryParameters.length > 0 && (
        <div className="canvas-modal__parameter-grid">
          {primaryParameters.map((parameter) => renderParameterField(parameter))}
        </div>
      )}
      {tuningParameters.length > 0 && (
        <div className="canvas-modal__parameter-grid" style={{ marginTop: '1rem' }}>
          {tuningParameters.map((parameter) => renderParameterField(parameter))}
        </div>
      )}
      <p className="canvas-modal__note canvas-modal__note--muted">
        {dropUnselected
          ? 'Unselected columns will be dropped from the dataset when this node runs.'
          : 'Unselected columns will remain in the dataset for downstream steps.'}
      </p>
      {selectedColumns.length > 0 ? (
        <p className="canvas-modal__note canvas-modal__note--info">
          Selected {selectedColumns.length.toLocaleString()} column(s): {selectedSummary}
        </p>
      ) : (
        <p className="canvas-modal__note">
          No columns selected yet. Run a preview to compute selection scores.
        </p>
      )}
      {droppedColumns.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Dropped {droppedColumns.length.toLocaleString()} column(s): {droppedSummary}
        </p>
      )}
      {summaries.length > 0 ? (
        <div className="canvas-cast__table-wrapper">
          <table className="canvas-cast__table">
            <thead>
              <tr>
                <th scope="col">Feature column</th>
                <th scope="col">Score</th>
                <th scope="col">p-value</th>
                <th scope="col">Rank</th>
                <th scope="col">Importance</th>
                <th scope="col">Status</th>
                <th scope="col">Note</th>
              </tr>
            </thead>
            <tbody>
              {summaries.map((summary) => {
                const rowClasses = ['canvas-cast__row'];
                if (!summary.selected) {
                  rowClasses.push('canvas-cast__row--muted');
                }
                return (
                  <tr key={`feature-selection-summary-${summary.column}`} className={rowClasses.join(' ')}>
                    <th scope="row">{summary.column}</th>
                    <td>{formatNumber(summary.score)}</td>
                    <td>{formatPValue(summary.p_value)}</td>
                    <td>{summary.rank ?? '—'}</td>
                    <td>{formatNumber(summary.importance, 4)}</td>
                    <td>{summary.selected ? 'Selected' : 'Dropped'}</td>
                    <td>{summary.note ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="canvas-modal__note">
          Preview results will display per-column scores, ranks, and selection outcomes here.
        </p>
      )}
      {notes.length > 0 && (
        <ul className="canvas-modal__note-list">
          {notes.map((note, index) => (
            <li key={`feature-selection-note-${index}`}>{note}</li>
          ))}
        </ul>
      )}
    </section>
  );
};
