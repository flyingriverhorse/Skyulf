import React, { useMemo, useEffect } from 'react';
import type {
  FeatureNodeParameter,
  HashEncodingColumnSuggestion,
  HashEncodingSuggestionStatus,
} from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import {
  extractPendingConfigurationDetails,
  type PendingConfigurationDetail,
} from '../../utils/pendingConfiguration';

type HashEncodingSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  isFetching: boolean;
  error: string | null;
  suggestions: HashEncodingColumnSuggestion[];
  metadata: {
    sampleSize: number | null;
    generatedAt: string | null;
    totalTextColumns: number;
    recommendedCount: number;
    autoDetectDefault: boolean;
    suggestedBucketDefault: number;
    highCardinalityColumns: string[];
    notes: string[];
  };
  columnsParameter: FeatureNodeParameter | null;
  autoDetectParameter: FeatureNodeParameter | null;
  autoDetectValue?: boolean;
  maxCategoriesParameter: FeatureNodeParameter | null;
  bucketsParameter: FeatureNodeParameter | null;
  outputSuffixParameter: FeatureNodeParameter | null;
  dropOriginalParameter: FeatureNodeParameter | null;
  encodeMissingParameter: FeatureNodeParameter | null;
  selectedColumns: string[];
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  onToggleColumn: (column: string) => void;
  onApplyRecommended: (columns: string[]) => void;
  formatMissingPercentage: (value?: number | null) => string;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (
    details: PendingConfigurationDetail[]
  ) => void;
  onPendingConfigurationCleared?: () => void;
};

const STATUS_LABELS: Record<HashEncodingSuggestionStatus, string> = {
  recommended: 'Recommended',
  high_cardinality: 'High cardinality',
  identifier: 'Identifier-like',
  free_text: 'Free text',
  single_category: 'Single category',
  too_many_categories: 'Too many categories',
};

const STATUS_BADGE_CLASS: Record<HashEncodingSuggestionStatus, string> = {
  recommended: 'canvas-cast__chip--applied',
  high_cardinality: 'canvas-cast__chip--attention',
  identifier: 'canvas-cast__chip--muted',
  free_text: 'canvas-cast__chip--muted',
  single_category: 'canvas-cast__chip--muted',
  too_many_categories: 'canvas-cast__chip--muted',
};

const formatMetric = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

export const HashEncodingSection: React.FC<HashEncodingSectionProps> = ({
  sourceId,
  hasReachableSource,
  isFetching,
  error,
  suggestions,
  metadata,
  columnsParameter,
  autoDetectParameter,
  autoDetectValue,
  maxCategoriesParameter,
  bucketsParameter,
  outputSuffixParameter,
  dropOriginalParameter,
  encodeMissingParameter,
  selectedColumns,
  renderParameterField,
  onToggleColumn,
  onApplyRecommended,
  formatMissingPercentage,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (!previewState?.data?.signals?.full_execution) {
      onPendingConfigurationCleared?.();
      return;
    }

    const details = extractPendingConfigurationDetails(
      previewState.data.signals.full_execution,
    );

    let relevantDetails = details;

    // If auto-detect is enabled, suppress "columns not configured" warnings
    if (autoDetectValue) {
      relevantDetails = relevantDetails.filter((d) => !d.label.toLowerCase().includes('columns'));
    }

    // If columns are manually selected, suppress "columns not configured" warnings
    if (selectedColumns && selectedColumns.length > 0) {
      relevantDetails = relevantDetails.filter((d) => !d.label.toLowerCase().includes('columns'));
    }

    if (relevantDetails.length > 0) {
      onPendingConfigurationWarning?.(relevantDetails);
    } else {
      onPendingConfigurationCleared?.();
    }
  }, [
    previewState?.data?.signals?.full_execution,
    onPendingConfigurationWarning,
    onPendingConfigurationCleared,
    autoDetectValue,
    selectedColumns,
  ]);

  const recommendedColumns = useMemo(() => {
    if (!suggestions.length) {
      return [] as string[];
    }
    const aggregate = new Set<string>();
    suggestions.forEach((entry) => {
      if (entry.status === 'recommended' && entry.selectable) {
        aggregate.add(entry.column);
      }
    });
    return Array.from(aggregate);
  }, [suggestions]);

  const connectionNote = useMemo(() => {
    if (!sourceId) {
      return 'Choose a dataset to surface hash encoding suggestions.';
    }
    if (!hasReachableSource) {
      return 'Connect this node to the dataset input to analyse categorical features.';
    }
    return null;
  }, [hasReachableSource, sourceId]);

  const hasSuggestions = suggestions.length > 0;
  const bucketDefaultNote = metadata.suggestedBucketDefault > 0
    ? `Default bucket suggestion: ${metadata.suggestedBucketDefault.toLocaleString()}.`
    : null;
  const showErrorMessage = !isFetching && Boolean(error) && Boolean(sourceId) && hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Hash encoding suggestions</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onApplyRecommended(recommendedColumns)}
            disabled={!recommendedColumns.length}
          >
            Add recommended
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Hash encoding maps categorical labels into a fixed number of numeric buckets, keeping feature width bounded even
        when categories grow over time. Collisions are acceptable for tree models and regularised linear models that can
        tolerate sparse overlaps.
      </p>
      {connectionNote && <p className="canvas-modal__note canvas-modal__note--warning">{connectionNote}</p>}
      {metadata.autoDetectDefault && (
        <p className="canvas-modal__note">
          Auto-detect is suggested so newly ingested categorical columns pick up hashing without manual edits.
        </p>
      )}
      {bucketDefaultNote && <p className="canvas-modal__note canvas-modal__note--info">{bucketDefaultNote}</p>}
      {metadata.generatedAt && (
        <p className="canvas-modal__note">
          Insights generated {new Date(metadata.generatedAt).toLocaleString()} on{' '}
          {metadata.sampleSize === null
            ? 'sampled rows'
            : `${metadata.sampleSize.toLocaleString()} row${metadata.sampleSize === 1 ? '' : 's'}`}.
        </p>
      )}
      {!metadata.generatedAt && metadata.sampleSize !== null && (
        <p className="canvas-modal__note">
          Evaluated {metadata.sampleSize.toLocaleString()} row{metadata.sampleSize === 1 ? '' : 's'} for categorical trends.
        </p>
      )}
      {metadata.totalTextColumns > 0 && (
        <p className="canvas-modal__note">
          Found {metadata.totalTextColumns} text column{metadata.totalTextColumns === 1 ? '' : 's'}; recommendations cover{' '}
          {metadata.recommendedCount} candidate{metadata.recommendedCount === 1 ? '' : 's'}.
        </p>
      )}
      {metadata.notes.length > 0 && (
        <ul className="canvas-modal__note-list">
          {metadata.notes.map((note, index) => (
            <li key={`hash-encoding-note-${index}`}>{note}</li>
          ))}
        </ul>
      )}
      {metadata.highCardinalityColumns.length > 0 && (
        <p className="canvas-modal__note">
          High-cardinality columns highlighted:{' '}
          {metadata.highCardinalityColumns.slice(0, 4).join(', ')}
          {metadata.highCardinalityColumns.length > 4 ? ', …' : ''}.
        </p>
      )}
      {isFetching && <p className="canvas-modal__note">Loading hash encoding insights…</p>}
      {showErrorMessage && <p className="canvas-modal__note canvas-modal__note--error">{error ?? ''}</p>}
      {!isFetching && !error && !hasSuggestions && sourceId && hasReachableSource && (
        <p className="canvas-modal__note">No categorical text candidates detected in the current preview.</p>
      )}
      {!isFetching && !error && hasSuggestions && (
        <div className="canvas-cast__table-wrapper">
          <table className="canvas-cast__table">
            <thead>
              <tr>
                <th scope="col">Column</th>
                <th scope="col">Distinct</th>
                <th scope="col">Missing</th>
                <th scope="col">Suggested buckets</th>
                <th scope="col">Status</th>
                <th scope="col">Select</th>
              </tr>
            </thead>
            <tbody>
              {suggestions.map((suggestion) => {
                const isSelected = selectedColumns.includes(suggestion.column);
                const statusLabel = STATUS_LABELS[suggestion.status];
                const statusClass = STATUS_BADGE_CLASS[suggestion.status] ?? 'canvas-cast__chip--muted';
                const sampleValues = suggestion.sample_values.length ? suggestion.sample_values.join(', ') : '—';
                const uniqueDisplay = suggestion.unique_count !== undefined && suggestion.unique_count !== null
                  ? `${formatMetric(suggestion.unique_count)} (${formatMetric(suggestion.unique_percentage)}%)`
                  : '—';
                const missingDisplay = formatMissingPercentage(suggestion.missing_percentage);
                const bucketDisplay = suggestion.recommended_bucket_count
                  ? suggestion.recommended_bucket_count.toLocaleString()
                  : '—';

                return (
                  <tr key={`hash-encoding-row-${suggestion.column}`} className="canvas-cast__row">
                    <th scope="row">
                      <div className="canvas-cast__column-cell">
                        <span className="canvas-cast__column-name">{suggestion.column}</span>
                        <div className="canvas-cast__column-meta">
                          {suggestion.dtype && <span className="canvas-cast__muted">dtype: {suggestion.dtype}</span>}
                          {suggestion.text_category && (
                            <span className="canvas-cast__badge canvas-cast__badge--category">
                              {suggestion.text_category}
                            </span>
                          )}
                        </div>
                        <div className="canvas-cast__column-meta">
                          <span className="canvas-cast__muted">Examples: {sampleValues}</span>
                        </div>
                        <p className="canvas-cast__recommendation-note">{suggestion.reason}</p>
                      </div>
                    </th>
                    <td>{uniqueDisplay}</td>
                    <td>{missingDisplay}</td>
                    <td>{bucketDisplay}</td>
                    <td>
                      <span className={`canvas-cast__chip ${statusClass}`}>{statusLabel}</span>
                    </td>
                    <td>
                      <label className="canvas-modal__checkbox-item">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => onToggleColumn(suggestion.column)}
                          disabled={!suggestion.selectable}
                        />
                        <span className="canvas-modal__checkbox-label">
                          {suggestion.selectable ? (isSelected ? 'In node' : 'Add column') : 'Not eligible'}
                        </span>
                      </label>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="canvas-modal__parameter-grid">
        {autoDetectParameter && renderParameterField(autoDetectParameter)}
        {maxCategoriesParameter && renderParameterField(maxCategoriesParameter)}
        {bucketsParameter && renderParameterField(bucketsParameter)}
        {outputSuffixParameter && renderParameterField(outputSuffixParameter)}
        {dropOriginalParameter && renderParameterField(dropOriginalParameter)}
        {encodeMissingParameter && renderParameterField(encodeMissingParameter)}
      </div>
      {columnsParameter && (
        <div>
          <h4 className="canvas-modal__subheading">Manual column selection</h4>
          {renderParameterField(columnsParameter)}
        </div>
      )}
    </section>
  );
};
