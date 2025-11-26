import React from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type {
  DropColumnCandidate,
  DropColumnRecommendationFilter,
} from '../../../../api';
import { useDropMissingColumns } from '../../hooks/useDropMissingColumns';

export { useDropMissingColumns };


type DropMissingColumnsSectionProps = {
  sourceId?: string | null;
  availableFilters: DropColumnRecommendationFilter[];
  activeFilterId: string | null;
  setActiveFilterId: Dispatch<SetStateAction<string | null>>;
  recommendations: DropColumnCandidate[];
  filteredRecommendations: DropColumnCandidate[];
  isFetchingRecommendations: boolean;
  recommendationsError: string | null;
  relativeGeneratedAt: string | null;
  formatSignalName: (signal?: string | null) => string | null;
  formatMissingPercentage: (value?: number | null) => string;
  getPriorityClass: (priority?: string | null) => string | null;
  getPriorityLabel: (priority?: string | null) => string | null;
  handleToggleColumn: (column: string) => void;
  selectedColumns: string[];
};

export const DropMissingColumnsSection: React.FC<DropMissingColumnsSectionProps> = ({
  sourceId,
  availableFilters,
  activeFilterId,
  setActiveFilterId,
  recommendations,
  filteredRecommendations,
  isFetchingRecommendations,
  recommendationsError,
  relativeGeneratedAt,
  formatSignalName,
  formatMissingPercentage,
  getPriorityClass,
  getPriorityLabel,
  handleToggleColumn,
  selectedColumns,
}) => {
  const renderRecommendationOption = (candidate: DropColumnCandidate) => {
    if (!candidate?.name) {
      return null;
    }

    const name = String(candidate.name);
    const checked = selectedColumns.includes(name);
    const priorityClass = getPriorityClass(candidate.priority);
    const priorityLabel = getPriorityLabel(candidate.priority);
    const signalLabels = Array.isArray(candidate?.signals)
      ? candidate.signals
          .map((signal) => formatSignalName(signal))
          .filter((label): label is string => Boolean(label))
      : [];

    return (
      <label key={name} className="canvas-modal__checkbox-item">
        <input type="checkbox" checked={checked} onChange={() => handleToggleColumn(name)} />
        <div className="canvas-modal__checkbox-content">
          <div className="canvas-modal__recommendation-name">{name}</div>
          <div className="canvas-modal__recommendation-meta">
            {candidate.missing_percentage !== undefined &&
              candidate.missing_percentage !== null && (
                <span>{`Missing: ${formatMissingPercentage(candidate.missing_percentage)}`}</span>
              )}
            {priorityClass && priorityLabel && (
              <span className={`canvas-modal__priority-badge canvas-modal__priority-badge--${priorityClass}`}>
                {priorityLabel}
              </span>
            )}
          </div>
          {signalLabels.length > 0 && (
            <div className="canvas-modal__signal-badges">
              {signalLabels.map((label) => (
                <span key={`${name}-${label}`} className="canvas-modal__signal-badge">
                  {label}
                </span>
              ))}
            </div>
          )}
          {candidate.reason && <p className="canvas-modal__recommendation-reason">{candidate.reason}</p>}
        </div>
      </label>
    );
  };

  if (!sourceId) {
    return (
      <p className="canvas-modal__note canvas-modal__note--warning">
        Choose a dataset to load EDA-backed column suggestions.
      </p>
    );
  }

  return (
    <>
      {availableFilters.length > 0 && (
        <div className="canvas-modal__filter-row" role="group" aria-label="Recommendation filters">
          <button
            type="button"
            className={`canvas-modal__filter-button${
              !activeFilterId ? ' canvas-modal__filter-button--active' : ''
            }`}
            onClick={() => setActiveFilterId(null)}
          >
            All reasons ({recommendations.length})
          </button>
          {availableFilters.map((filter) => {
            const isActive = activeFilterId === filter.id;
            return (
              <button
                key={filter.id}
                type="button"
                className={`canvas-modal__filter-button${
                  isActive ? ' canvas-modal__filter-button--active' : ''
                }`}
                onClick={() =>
                  setActiveFilterId((current) => (current === filter.id ? null : filter.id))
                }
                title={filter.description ?? undefined}
              >
                {filter.label} ({filter.count})
              </button>
            );
          })}
        </div>
      )}
      {isFetchingRecommendations && (
        <p className="canvas-modal__note">Loading column recommendations…</p>
      )}
      {!isFetchingRecommendations && recommendationsError && (
        <p className="canvas-modal__note canvas-modal__note--error">{recommendationsError}</p>
      )}
      {!isFetchingRecommendations && !recommendationsError && (
        <>
          {relativeGeneratedAt && (
            <p className="canvas-modal__note">Recommendations generated {relativeGeneratedAt}.</p>
          )}
          {filteredRecommendations.length ? (
            <div className="canvas-modal__checkbox-list" role="group" aria-label="Recommended columns">
              {filteredRecommendations.map((candidate) => renderRecommendationOption(candidate))}
            </div>
          ) : (
            <p className="canvas-modal__note">
              {activeFilterId
                ? 'No columns matched this filter in the latest quality scan.'
                : 'No columns crossed the missingness threshold in the latest quality scan.'}
            </p>
          )}
        </>
      )}
    </>
  );
};
