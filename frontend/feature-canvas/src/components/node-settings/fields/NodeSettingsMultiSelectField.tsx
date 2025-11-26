import React from 'react';
import { FeatureNodeParameter } from '../../../api';
import { DropMissingColumnsSection } from '../nodes/drop_col_rows/DropMissingColumnsSection';
import {
  formatMetricValue,
  formatMissingPercentage,
  formatNumericStat,
  getPriorityClass,
  getPriorityLabel,
} from '../formatting';
import { formatColumnType } from '../utils/formatters';

type NodeSettingsMultiSelectFieldProps = {
  parameter: FeatureNodeParameter;
  previewStateStatus: string;
  isBinningNode: boolean;
  isScalingNode: boolean;
  binningAllNumericColumns: string[];
  binningRecommendedColumnSet: Set<string>;
  selectedColumns: string[];
  availableColumns: string[];
  scalingExcludedColumns: Set<string>;
  normalizedColumnSearch: string;
  filteredColumnOptions: string[];
  binningExcludedColumns: Set<string>;
  selectionCount: number;
  isCastNode: boolean;
  columnSuggestions: Record<string, string[]>;
  sourceId?: string | null;
  isFetchingRecommendations: boolean;
  hasReachableSource: boolean;
  refreshRecommendations: () => void;
  availableFilters: any[];
  activeFilterId: string | null;
  setActiveFilterId: React.Dispatch<React.SetStateAction<string | null>>;
  recommendations: any[];
  filteredRecommendations: any[];
  recommendationsError: string | null;
  relativeGeneratedAt: string | null;
  formatSignalName: (signal?: string | null | undefined) => string | null;
  handleToggleColumn: (column: string) => void;
  handleRemoveColumn: (column: string) => void;
  handleApplyAllRecommended: () => void;
  handleBinningApplyColumns: (columns: Set<string>) => void;
  handleSelectAllColumns: () => void;
  handleClearColumns: () => void;
  columnSearch: string;
  setColumnSearch: (search: string) => void;
  columnMissingMap: Record<string, number>;
  columnTypeMap: Record<string, string>;
  binningColumnPreviewMap: Record<string, any>;
  isImputerNode: boolean;
  showRecommendations: boolean;
};

export const NodeSettingsMultiSelectField: React.FC<NodeSettingsMultiSelectFieldProps> = ({
  parameter,
  previewStateStatus,
  isBinningNode,
  isScalingNode,
  binningAllNumericColumns,
  binningRecommendedColumnSet,
  selectedColumns,
  availableColumns,
  scalingExcludedColumns,
  normalizedColumnSearch,
  filteredColumnOptions,
  binningExcludedColumns,
  selectionCount,
  isCastNode,
  columnSuggestions,
  sourceId,
  isFetchingRecommendations,
  hasReachableSource,
  refreshRecommendations,
  availableFilters,
  activeFilterId,
  setActiveFilterId,
  recommendations,
  filteredRecommendations,
  recommendationsError,
  relativeGeneratedAt,
  formatSignalName,
  handleToggleColumn,
  handleRemoveColumn,
  handleApplyAllRecommended,
  handleBinningApplyColumns,
  handleSelectAllColumns,
  handleClearColumns,
  columnSearch,
  setColumnSearch,
  columnMissingMap,
  columnTypeMap,
  binningColumnPreviewMap,
  isImputerNode,
  showRecommendations,
}) => {
  const requiresRecommendations = parameter?.source?.type === 'drop_column_recommendations';
  const isCatalogOnly = !requiresRecommendations;
  const isCatalogLoading = isCatalogOnly && previewStateStatus === 'loading';
  const isBinningColumnsParameter = isBinningNode && parameter.name === 'columns';
  const isScalingColumnsParameter = isScalingNode && parameter.name === 'columns';

  const binningCandidateColumns = (() => {
    if (!isBinningColumnsParameter) {
      return [] as string[];
    }
    const merged = [...binningAllNumericColumns];
    if (binningRecommendedColumnSet.size > 0) {
      binningRecommendedColumnSet.forEach((column) => {
        if (!merged.includes(column)) {
          merged.push(column);
        }
      });
    }
    selectedColumns.forEach((column) => {
      if (column && !merged.includes(column)) {
        merged.push(column);
      }
    });
    return merged;
  })();

  const displayedColumns = (() => {
    if (isBinningColumnsParameter) {
      return binningCandidateColumns;
    }
    if (isScalingColumnsParameter) {
      return availableColumns.filter((column) => !scalingExcludedColumns.has(column));
    }
    return availableColumns;
  })();

  const renderedColumnOptions = (() => {
    if (isBinningColumnsParameter) {
      if (!normalizedColumnSearch) {
        return binningCandidateColumns;
      }
      return binningCandidateColumns.filter((column) =>
        column.toLowerCase().includes(normalizedColumnSearch)
      );
    }
    if (isScalingColumnsParameter) {
      return filteredColumnOptions.filter((column) => !scalingExcludedColumns.has(column));
    }
    return filteredColumnOptions;
  })();

  const selectionDisplayCount = isBinningColumnsParameter
    ? selectedColumns.filter((column) => !binningExcludedColumns.has(column)).length
    : isScalingColumnsParameter
      ? selectedColumns.filter((column) => !scalingExcludedColumns.has(column)).length
      : selectionCount;

  const allColumnsSelected =
    isCatalogOnly && displayedColumns.length > 0 && selectionDisplayCount >= displayedColumns.length;
  const showMissingMetric = requiresRecommendations || isImputerNode;
  const availableColumnSet = new Set(displayedColumns);
  const suggestionSummaries =
    isCastNode && isCatalogOnly
      ? Object.entries(columnSuggestions)
          .filter(([name, suggestions]) => availableColumnSet.has(name) && suggestions.length > 0)
          .map(([name, suggestions]) => `${name}: ${suggestions.join(', ')}`)
          .slice(0, 4)
      : [];
  const binningExcludedPreview = isBinningColumnsParameter ? Array.from(binningExcludedColumns).slice(0, 4) : [];
  const scalingExcludedPreview = isScalingColumnsParameter ? Array.from(scalingExcludedColumns).slice(0, 4) : [];
  const hasBackendRecommendations = isBinningColumnsParameter && binningRecommendedColumnSet.size > 0;
  const canAddRecommendedColumns = hasBackendRecommendations
    ? Array.from(binningRecommendedColumnSet).some((column) => !selectedColumns.includes(column))
    : false;

  return (
    <div
      key={parameter.name}
      className="canvas-modal__parameter-field canvas-modal__parameter-field--multiselect"
    >
      <div className="canvas-modal__parameter-label">
        <span>{parameter.label}</span>
        {requiresRecommendations && sourceId && (
          <div className="canvas-modal__parameter-actions">
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={refreshRecommendations}
              disabled={isFetchingRecommendations || !hasReachableSource}
            >
              {isFetchingRecommendations ? 'Refreshing…' : 'Refresh suggestions'}
            </button>
          </div>
        )}
      </div>
      {parameter.description && (
        <p className="canvas-modal__parameter-description">{parameter.description}</p>
      )}

      {suggestionSummaries.length > 0 && (
        <p className="canvas-modal__note">
          Smart suggestions: {suggestionSummaries.join('; ')}
        </p>
      )}

      {isBinningColumnsParameter && binningExcludedColumns.size > 0 && (
        <p className="canvas-modal__note">
          Skipping {binningExcludedColumns.size} non-numeric column
          {binningExcludedColumns.size === 1 ? '' : 's'}
          {binningExcludedPreview.length
            ? ` (examples: ${binningExcludedPreview.join(', ')}${binningExcludedColumns.size > binningExcludedPreview.length ? ', …' : ''})`
            : ''}
          — binning only supports numeric inputs.
        </p>
      )}

      {isScalingColumnsParameter && scalingExcludedColumns.size > 0 && (
        <p className="canvas-modal__note">
          Skipping {scalingExcludedColumns.size} non-numeric column
          {scalingExcludedColumns.size === 1 ? '' : 's'}
          {scalingExcludedPreview.length
            ? ` (examples: ${scalingExcludedPreview.join(', ')}${scalingExcludedColumns.size > scalingExcludedPreview.length ? ', …' : ''})`
            : ''}
          — scaling only supports numeric inputs.
        </p>
      )}

      {requiresRecommendations && (
        <DropMissingColumnsSection
          sourceId={sourceId}
          availableFilters={availableFilters}
          activeFilterId={activeFilterId}
          setActiveFilterId={setActiveFilterId}
          recommendations={recommendations}
          filteredRecommendations={filteredRecommendations}
          isFetchingRecommendations={isFetchingRecommendations}
          recommendationsError={recommendationsError}
          relativeGeneratedAt={relativeGeneratedAt}
          formatSignalName={formatSignalName}
          formatMissingPercentage={formatMissingPercentage}
          getPriorityClass={getPriorityClass}
          getPriorityLabel={getPriorityLabel}
          handleToggleColumn={handleToggleColumn}
          selectedColumns={selectedColumns}
        />
      )}

      <div className="canvas-modal__note">
        Selected columns: <strong>{selectionDisplayCount}</strong>
      </div>

      {selectionDisplayCount > 0 && (
        <div className="canvas-modal__selection-summary">
          <h4>Selected columns</h4>
          <div className="canvas-modal__selection-chips">
            {selectedColumns.map((column) => (
              <span key={column} className="canvas-modal__selection-chip">
                {column}
                <button
                  type="button"
                  onClick={() => handleRemoveColumn(column)}
                  aria-label={`Remove ${column}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="canvas-modal__multi-select-actions">
        {requiresRecommendations && (
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleApplyAllRecommended}
            disabled={!recommendations.length}
          >
            Use all recommendations
          </button>
        )}
        {hasBackendRecommendations && (
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={() => handleBinningApplyColumns(binningRecommendedColumnSet)}
            disabled={!canAddRecommendedColumns}
          >
            Add recommended columns
          </button>
        )}
        {isCatalogOnly && (
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleSelectAllColumns}
            disabled={!displayedColumns.length || allColumnsSelected}
          >
            Select all columns
          </button>
        )}
        <button
          type="button"
          className="btn btn-outline-secondary"
          onClick={handleClearColumns}
          disabled={!selectionCount}
        >
          Clear selection
        </button>
      </div>

      <div className="canvas-modal__all-columns">
        <h4>All columns</h4>
        <div className="canvas-modal__all-columns-search">
          <input
            type="text"
            className="canvas-modal__custom-input"
            value={columnSearch}
            onChange={(event) => setColumnSearch(event.target.value)}
            placeholder="Search columns"
            aria-label="Search columns"
          />
        </div>
        <div className="canvas-modal__all-columns-list" role="group" aria-label="All columns">
          {renderedColumnOptions.length ? (
            renderedColumnOptions.map((column) => {
              const isSelected = selectedColumns.includes(column);
              const missingValue = Object.prototype.hasOwnProperty.call(columnMissingMap, column)
                ? columnMissingMap[column]
                : undefined;
              const missingLabel = formatMissingPercentage(
                typeof missingValue === 'number' ? missingValue : null
              );
              const columnType = columnTypeMap[column] ?? null;
              const columnSuggestionList = columnSuggestions[column] ?? [];
              const rangeMeta: { min: number | null; max: number | null; distinct?: number | null } | undefined =
                isBinningColumnsParameter
                  ? binningColumnPreviewMap[column]
                  : undefined;
              const hasRangeMeta = Boolean(
                rangeMeta &&
                  ((rangeMeta.min !== null && rangeMeta.min !== undefined) ||
                    (rangeMeta.max !== null && rangeMeta.max !== undefined))
              );
              const distinctCount = isBinningColumnsParameter ? rangeMeta?.distinct ?? null : null;
              const hasDistinctMeta =
                isBinningColumnsParameter && distinctCount !== null && Number.isFinite(distinctCount);
              const distinctDisplayValue = hasDistinctMeta ? (distinctCount as number) : null;
              const showTypeMeta = !isBinningColumnsParameter;
              const shouldShowMetaRow = showTypeMeta || showMissingMetric || hasRangeMeta || hasDistinctMeta;
              const hasSuggestionHints = showTypeMeta && columnSuggestionList.length > 0;
              return (
                <label
                  key={column}
                  className={`canvas-modal__checkbox-item canvas-modal__checkbox-item--compact${
                    isSelected ? ' canvas-modal__checkbox-item--selected' : ''
                  }`}
                >
                  <input type="checkbox" checked={isSelected} onChange={() => handleToggleColumn(column)} />
                  <div className="canvas-modal__column-option">
                    <span className="canvas-modal__column-option-name">{column}</span>
                    {shouldShowMetaRow && (
                      <div className="canvas-modal__column-option-meta">
                        {showTypeMeta && (
                          <span className="canvas-modal__column-option-metric">
                            Type: {formatColumnType(columnType)}
                          </span>
                        )}
                        {showMissingMetric && (
                          <span className="canvas-modal__column-option-metric">Missing: {missingLabel}</span>
                        )}
                        {hasRangeMeta && (
                          <span className="canvas-modal__column-option-metric">
                            {isBinningColumnsParameter ? 'Sample range' : 'Range'}:{' '}
                            {rangeMeta && rangeMeta.min !== null && rangeMeta.min !== undefined
                              ? formatNumericStat(rangeMeta.min)
                              : '—'}{' '}
                            –{' '}
                            {rangeMeta && rangeMeta.max !== null && rangeMeta.max !== undefined
                              ? formatNumericStat(rangeMeta.max)
                              : '—'}
                          </span>
                        )}
                        {hasDistinctMeta && distinctDisplayValue !== null && (
                          <span className="canvas-modal__column-option-metric">
                            Distinct sample values: {formatMetricValue(distinctDisplayValue)}
                          </span>
                        )}
                      </div>
                    )}
                    {hasSuggestionHints && (
                      <span className="canvas-modal__column-option-hint">
                        Suggested: {columnSuggestionList.join(', ')}
                      </span>
                    )}
                  </div>
                </label>
              );
            })
          ) : (
            <p className="canvas-modal__note">
              {availableColumns.length
                ? isBinningColumnsParameter && displayedColumns.length === 0
                  ? 'No eligible columns for binning (non-numeric fields are skipped).'
                  : 'No columns match your search.'
                : isCatalogLoading
                  ? 'Loading column catalog…'
                  : !sourceId
                    ? 'Select a dataset to load column catalog.'
                    : !hasReachableSource
                      ? 'Connect this step to an upstream output to load column catalog.'
                      : 'Column catalog unavailable for this dataset.'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
