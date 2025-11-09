import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type {
  DropColumnCandidate,
  DropColumnRecommendationFilter,
} from '../../../../api';
import { fetchDropColumnRecommendations } from '../../../../api';

export type DropMissingColumnsArgs = {
  hasDropColumnSource: boolean;
  sourceId?: string | null;
  nodeId?: string | null;
  graphContext?: { nodes: any[]; edges: any[] } | null;
  hasReachableSource: boolean;
  columnTypeMap: Record<string, string>;
  setAvailableColumns: Dispatch<SetStateAction<string[]>>;
  setColumnSearch: Dispatch<SetStateAction<string>>;
  setColumnMissingMap: Dispatch<SetStateAction<Record<string, number>>>;
  setColumnTypeMap: Dispatch<SetStateAction<Record<string, string>>>;
  setColumnSuggestions: Dispatch<SetStateAction<Record<string, string[]>>>;
};

export type DropMissingColumnsState = {
  hasDropColumnSource: boolean;
  availableFilters: DropColumnRecommendationFilter[];
  activeFilterId: string | null;
  setActiveFilterId: Dispatch<SetStateAction<string | null>>;
  recommendations: DropColumnCandidate[];
  filteredRecommendations: DropColumnCandidate[];
  formatSignalName: (signal?: string | null) => string | null;
  isFetchingRecommendations: boolean;
  recommendationsError: string | null;
  recommendationsGeneratedAt: string | null;
  suggestedThreshold: number | null;
  refreshRecommendations: () => void;
};

export const useDropMissingColumns = ({
  hasDropColumnSource,
  sourceId,
  nodeId,
  graphContext,
  hasReachableSource,
  columnTypeMap,
  setAvailableColumns,
  setColumnSearch,
  setColumnMissingMap,
  setColumnTypeMap,
  setColumnSuggestions,
}: DropMissingColumnsArgs): DropMissingColumnsState => {
  const [availableFilters, setAvailableFilters] = useState<DropColumnRecommendationFilter[]>([]);
  const [activeFilterId, setActiveFilterId] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<DropColumnCandidate[]>([]);
  const [recommendationsError, setRecommendationsError] = useState<string | null>(null);
  const [isFetchingRecommendations, setIsFetchingRecommendations] = useState(false);
  const [recommendationsGeneratedAt, setRecommendationsGeneratedAt] = useState<string | null>(null);
  const [suggestedThreshold, setSuggestedThreshold] = useState<number | null>(null);
  const [requestId, setRequestId] = useState(0);

  const columnTypeMapRef = useRef(columnTypeMap);
  useEffect(() => {
    columnTypeMapRef.current = columnTypeMap;
  }, [columnTypeMap]);

  useEffect(() => {
    let isActive = true;

    const resetState = (message?: string | null) => {
      setRecommendations([]);
      setRecommendationsError(message ?? null);
      setRecommendationsGeneratedAt(null);
      setSuggestedThreshold(null);
      setIsFetchingRecommendations(false);
      setAvailableFilters([]);
      setActiveFilterId(null);
      setAvailableColumns([]);
      setColumnSearch('');
      setColumnMissingMap({});
      setColumnTypeMap({});
      setColumnSuggestions({});
    };

    if (!hasDropColumnSource) {
      resetState();
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      resetState();
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      resetState('Connect this step to an upstream output to load recommendations.');
      return () => {
        isActive = false;
      };
    }

    setIsFetchingRecommendations(true);
    setRecommendationsError(null);

    fetchDropColumnRecommendations(sourceId, {
      graph: graphContext,
      targetNodeId: nodeId ?? null,
    })
      .then((result) => {
        if (!isActive) {
          return;
        }

        const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
        setRecommendations(candidates);
        setRecommendationsGeneratedAt(result?.generated_at ?? null);
        setSuggestedThreshold(
          typeof result?.suggested_threshold === 'number' ? Number(result.suggested_threshold) : null,
        );

        const filters = Array.isArray(result?.available_filters) ? result.available_filters : [];
        setAvailableFilters(filters);
        setActiveFilterId((previous) =>
          previous && filters.some((filter) => filter.id === previous) ? previous : null,
        );

        const sanitizedMissingMap: Record<string, number> = {};
        if (
          result &&
          typeof result === 'object' &&
          result?.column_missing_map &&
          typeof result.column_missing_map === 'object'
        ) {
          Object.entries(result.column_missing_map).forEach(([key, rawValue]) => {
            const trimmedKey = String(key ?? '').trim();
            if (!trimmedKey) {
              return;
            }
            const numericValue = Number(rawValue);
            if (!Number.isFinite(numericValue)) {
              return;
            }
            sanitizedMissingMap[trimmedKey] = Number(numericValue);
          });
        }

        const columnSet = new Set<string>();
        if (Array.isArray(result?.all_columns)) {
          result.all_columns.forEach((column) => {
            const trimmed = String(column ?? '').trim();
            if (trimmed) {
              columnSet.add(trimmed);
            }
          });
        }
        Object.keys(columnTypeMapRef.current ?? {}).forEach((column) => columnSet.add(column));

        const orderedColumns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));
        orderedColumns.forEach((column) => {
          if (!Object.prototype.hasOwnProperty.call(sanitizedMissingMap, column)) {
            sanitizedMissingMap[column] = 0;
          }
        });

        setAvailableColumns(orderedColumns);
        setColumnSearch('');
        setColumnMissingMap(sanitizedMissingMap);
        setColumnTypeMap((previous) => {
          const columnTypePayload = (result as any)?.column_type_map;
          if (!columnTypePayload || typeof columnTypePayload !== 'object') {
            return previous;
          }
          const next = { ...previous } as Record<string, string>;
          Object.entries(columnTypePayload as Record<string, any>).forEach(([key, value]) => {
            const trimmedKey = String(key ?? '').trim();
            if (!trimmedKey) {
              return;
            }
            const trimmedValue = String(value ?? '').trim();
            next[trimmedKey] = trimmedValue || previous[trimmedKey] || '';
          });
          return next;
        });

        setColumnSuggestions((previous) => {
          const columnRecommendationPayload = (result as any)?.column_recommendations;
          if (!columnRecommendationPayload || typeof columnRecommendationPayload !== 'object') {
            return previous;
          }
          const next: Record<string, string[]> = {};
          Object.entries(columnRecommendationPayload as Record<string, any>).forEach(([key, values]) => {
            const trimmedKey = String(key ?? '').trim();
            if (!trimmedKey) {
              return;
            }
            const list = Array.isArray(values)
              ? values.map((value) => String(value ?? '').trim()).filter(Boolean)
              : [];
            next[trimmedKey] = list;
          });
          return next;
        });
      })
      .catch((error: any) => {
        if (!isActive) {
          return;
        }
        resetState(error?.message ?? 'Unable to load recommendations');
      })
      .finally(() => {
        if (isActive) {
          setIsFetchingRecommendations(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [
    columnTypeMapRef,
    graphContext,
    hasDropColumnSource,
    hasReachableSource,
    nodeId,
    requestId,
    setAvailableColumns,
    setColumnMissingMap,
    setColumnSearch,
    setColumnSuggestions,
    setColumnTypeMap,
    sourceId,
  ]);

  const refreshRecommendations = useCallback(() => {
    if (!hasDropColumnSource || !sourceId || !hasReachableSource) {
      return;
    }
    setRequestId((value) => value + 1);
  }, [hasDropColumnSource, hasReachableSource, sourceId]);

  const signalLabelMap = useMemo(() => {
    const entries = new Map<string, string>();
    availableFilters.forEach((filter) => {
      if (filter?.id) {
        entries.set(filter.id, filter.label);
      }
    });
    return entries;
  }, [availableFilters]);

  const formatSignalName = useCallback(
    (signal?: string | null) => {
      if (!signal) {
        return null;
      }
      if (signalLabelMap.has(signal)) {
        return signalLabelMap.get(signal) ?? null;
      }
      const normalized = signal
        .split('_')
        .filter(Boolean)
        .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
        .join(' ');
      return normalized || signal;
    },
    [signalLabelMap],
  );

  const filteredRecommendations = useMemo(() => {
    if (!activeFilterId) {
      return recommendations;
    }
    return recommendations.filter((candidate) =>
      Array.isArray(candidate?.signals) && candidate.signals.includes(activeFilterId),
    );
  }, [activeFilterId, recommendations]);

  return {
    hasDropColumnSource,
    availableFilters,
    activeFilterId,
    setActiveFilterId,
    recommendations,
    filteredRecommendations,
    formatSignalName,
    isFetchingRecommendations,
    recommendationsError,
    recommendationsGeneratedAt,
    suggestedThreshold,
    refreshRecommendations,
  };
};

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
