import React, { useEffect } from 'react';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';
import { FormField, SelectInput, NumberInput, TextInput, RangeInput } from '../../ui/FormFields';
import {
  IMPUTER_MISSING_FILTER_PRESETS,
  isLikelyNumericColumn,
  isNumericImputationMethod,
  type ImputationMethodOption,
  type ImputationStrategyConfig,
  type ImputationStrategyMethod,
  type ImputerColumnOption,
} from './imputationSettings';

type ImputationSchemaDetail = {
  name: string;
  logical_family: string;
};

export type ImputationSchemaDiagnostics = {
  blocked: boolean;
  message: string;
  entries: Array<{
    index: number;
    columns: string[];
    details: ImputationSchemaDetail[];
  }>;
};

type ImputationStrategiesSectionProps = {
  hasSource: boolean;
  hasReachableSource: boolean;
  strategies: ImputationStrategyConfig[];
  columnOptions: ImputerColumnOption[];
  methodOptions: ImputationMethodOption[];
  missingFilter: number;
  missingFilterMax: number;
  missingFilterActive: boolean;
  filteredOptionCount: number;
  collapsedStrategies: Set<number>;
  onToggleStrategy: (index: number) => void;
  onRemoveStrategy: (index: number) => void;
  onAddStrategy: () => void;
  onMethodChange: (index: number, method: ImputationStrategyMethod) => void;
  onOptionNumberChange: (index: number, key: 'neighbors' | 'max_iter', value: string) => void;
  onColumnsChange: (index: number, value: string) => void;
  onColumnToggle: (index: number, column: string) => void;
  onMissingFilterChange: (value: number) => void;
  formatMissingPercentage: (value: number | null | undefined) => string;
  formatNumericStat: (value: number | null | undefined) => string;
  formatModeStat: (value: string | number | null | undefined) => string;
  schemaDiagnostics?: ImputationSchemaDiagnostics | null;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const ImputationStrategiesSection: React.FC<ImputationStrategiesSectionProps> = ({
  hasSource,
  hasReachableSource,
  strategies,
  columnOptions,
  methodOptions,
  missingFilter,
  missingFilterMax,
  missingFilterActive,
  filteredOptionCount,
  collapsedStrategies,
  onToggleStrategy,
  onRemoveStrategy,
  onAddStrategy,
  onMethodChange,
  onOptionNumberChange,
  onColumnsChange,
  onColumnToggle,
  onMissingFilterChange,
  formatMissingPercentage,
  formatNumericStat,
  formatModeStat,
  schemaDiagnostics,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      if (details.length > 0) {
        onPendingConfigurationWarning?.(details);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared]);

  const schemaDiagnosticsMap = React.useMemo(() => {
    if (!schemaDiagnostics?.entries?.length) {
      return new Map<number, ImputationSchemaDiagnostics['entries'][number]>();
    }
    return new Map(schemaDiagnostics.entries.map((entry) => [entry.index, entry]));
  }, [schemaDiagnostics]);
  const sliderMax = Math.max(missingFilterMax, missingFilter || 0);
  const hasCatalog = columnOptions.length > 0;

  const renderMethodSummary = (strategy: ImputationStrategyConfig) => {
    const baseLabel = methodOptions.find((option) => option.value === strategy.method)?.label ?? strategy.method;
    if (strategy.method === 'knn' && strategy.options?.neighbors) {
      return `${baseLabel} · k=${strategy.options.neighbors}`;
    }
    if ((strategy.method === 'regression' || strategy.method === 'mice') && strategy.options?.max_iter) {
      return `${baseLabel} · ${strategy.options.max_iter} iter`;
    }
    return baseLabel;
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Imputation strategies</h3>
        <div className="canvas-modal__section-actions">
          <button type="button" className="btn btn-outline-secondary" onClick={onAddStrategy}>
            Add strategy
          </button>
        </div>
      </div>
      {!hasSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to surface missingness insights and recommended columns.
        </p>
      )}
      {hasSource && !hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this step to an upstream output to load missingness insights.
        </p>
      )}
      {schemaDiagnostics?.blocked && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {schemaDiagnostics.message}
        </p>
      )}
      {columnOptions.length > 0 && (
        <div className="canvas-imputer__filter" role="group" aria-label="Filter columns by missing percentage">
          <div className="canvas-imputer__filter-top">
            <label htmlFor="imputer-missing-filter">Minimum missing percentage</label>
            <span className="canvas-imputer__filter-summary">
              Showing {filteredOptionCount} of {columnOptions.length} column
              {columnOptions.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="canvas-imputer__filter-controls">
            <RangeInput
              id="imputer-missing-filter"
              min={0}
              max={sliderMax}
              step={1}
              value={missingFilter}
              onChange={(event) => onMissingFilterChange(Number(event.target.value))}
            />
            <span className="canvas-imputer__filter-value">
              ≥<strong>{missingFilter}%</strong> missing
            </span>
            <button
              type="button"
              className="canvas-imputer__filter-reset"
              onClick={() => onMissingFilterChange(0)}
              disabled={!missingFilter}
            >
              Clear
            </button>
          </div>
          <div className="canvas-imputer__filter-presets">
            {IMPUTER_MISSING_FILTER_PRESETS.map((preset) => (
              <button
                key={`imputer-missing-preset-${preset}`}
                type="button"
                className={`canvas-imputer__filter-button${
                  missingFilter === preset ? ' canvas-imputer__filter-button--active' : ''
                }`}
                onClick={() => onMissingFilterChange(preset)}
              >
                ≥{preset}%
              </button>
            ))}
          </div>
        </div>
      )}
      {strategies.length ? (
        <div className="canvas-imputer__list">
          {strategies.map((strategy, index) => {
            const columnsValue = strategy.columns.join(', ');
            const requiresNumericColumns = isNumericImputationMethod(strategy.method);
            const columnOptionsForStrategy = columnOptions.filter((option) => {
              if (strategy.columns.includes(option.name)) {
                return true;
              }
              if (requiresNumericColumns && !isLikelyNumericColumn(option)) {
                return false;
              }
              const missingValue = typeof option.missingPercentage === 'number' ? option.missingPercentage : null;
              if (missingValue !== null) {
                if (missingValue <= 0) {
                  return false;
                }
                return missingValue >= missingFilter;
              }
              return missingFilter === 0;
            });
            const methodSummary = renderMethodSummary(strategy);
            const columnSummary = strategy.columns.length
              ? `${strategy.columns.length} column${strategy.columns.length === 1 ? '' : 's'}`
              : 'No columns yet';
            const isCollapsed = collapsedStrategies.has(index);
            const selectAllCandidates = columnOptionsForStrategy.filter(
              (option) => !strategy.columns.includes(option.name),
            );
            const dtypeIssues = schemaDiagnosticsMap.get(index);
            const dtypeSummary = dtypeIssues?.details?.length
              ? dtypeIssues.details
                  .map((detail: ImputationSchemaDetail) =>
                    detail.logical_family ? `${detail.name} (${detail.logical_family})` : detail.name,
                  )
                  .join(', ')
              : '';

            const handleSelectAllColumns = () => {
              if (!columnOptionsForStrategy.length) {
                return;
              }
              const combined = [...strategy.columns];
              columnOptionsForStrategy.forEach((option) => {
                if (!combined.includes(option.name)) {
                  combined.push(option.name);
                }
              });
              onColumnsChange(index, combined.join(', '));
            };

            return (
              <div key={`imputer-strategy-${index}`} className="canvas-imputer__card">
                <div className="canvas-imputer__card-header">
                  <button
                    type="button"
                    className="canvas-imputer__card-toggle"
                    onClick={() => onToggleStrategy(index)}
                    aria-expanded={!isCollapsed}
                    aria-controls={`imputer-strategy-body-${index}`}
                  >
                    <span
                      className={`canvas-imputer__toggle-icon${
                        isCollapsed ? '' : ' canvas-imputer__toggle-icon--open'
                      }`}
                      aria-hidden="true"
                    />
                    <span className="canvas-imputer__card-text">
                      <span className="canvas-imputer__card-title">Strategy {index + 1}</span>
                      <span className="canvas-imputer__card-summary">{methodSummary} · {columnSummary}</span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="canvas-imputer__remove"
                    onClick={() => onRemoveStrategy(index)}
                    aria-label={`Remove strategy ${index + 1}`}
                  >
                    Remove
                  </button>
                </div>
                {!isCollapsed && (
                  <div className="canvas-imputer__card-body" id={`imputer-strategy-body-${index}`}>
                    <FormField label="Method">
                      <SelectInput
                        value={strategy.method}
                        onChange={(event) =>
                          onMethodChange(index, (event.target.value as ImputationStrategyMethod) ?? 'mean')
                        }
                        options={methodOptions}
                      />
                    </FormField>
                    {strategy.method === 'knn' && (
                      <FormField label="Neighbors (k)">
                        <NumberInput
                          min={1}
                          max={50}
                          step={1}
                          value={strategy.options?.neighbors ?? 5}
                          onChange={(event) => onOptionNumberChange(index, 'neighbors', event.target.value)}
                        />
                      </FormField>
                    )}
                    {(strategy.method === 'regression' || strategy.method === 'mice') && (
                      <FormField label="Max iterations">
                        <NumberInput
                          min={1}
                          max={100}
                          step={1}
                          value={strategy.options?.max_iter ?? 10}
                          onChange={(event) => onOptionNumberChange(index, 'max_iter', event.target.value)}
                        />
                      </FormField>
                    )}
                    {strategy.method === 'mice' && (
                      <p className="canvas-modal__note">
                        MICE performs chained regressions with posterior sampling for robust imputations.
                      </p>
                    )}
                    <div className="canvas-imputer__row">
                      <label>Columns</label>
                      {dtypeSummary && (
                        <p className="canvas-modal__note canvas-modal__note--warning">
                          Remove or recast columns before applying this strategy: {dtypeSummary}.
                        </p>
                      )}
                      {columnOptionsForStrategy.length > 0 && selectAllCandidates.length > 0 && (
                        <button
                          type="button"
                          className="btn btn-link"
                          onClick={handleSelectAllColumns}
                        >
                          Select all columns
                        </button>
                      )}
                      {columnOptionsForStrategy.length ? (
                        <div className="canvas-imputer__columns-list" role="listbox" aria-label="Columns eligible for imputation">
                          {columnOptionsForStrategy.map((option) => {
                            const isSelected = strategy.columns.includes(option.name);
                            const missingLabel = formatMissingPercentage(option.missingPercentage);
                            const hasMean = typeof option.mean === 'number';
                            const hasMedian = typeof option.median === 'number';
                            const hasMode = option.mode !== null && option.mode !== undefined && option.mode !== '';
                            const showStats = hasMean || hasMedian || hasMode;
                            return (
                              <button
                                key={option.name}
                                type="button"
                                className={`canvas-imputer__column-pill${
                                  isSelected ? ' canvas-imputer__column-pill--selected' : ''
                                }`}
                                onClick={() => onColumnToggle(index, option.name)}
                                aria-pressed={isSelected}
                              >
                                <span className="canvas-imputer__column-pill-name">{option.name}</span>
                                <div className="canvas-imputer__column-pill-meta">
                                  <span className="canvas-imputer__column-pill-metric">Missing: {missingLabel}</span>
                                  {option.dtype && (
                                    <span className="canvas-imputer__column-pill-dtype">{option.dtype}</span>
                                  )}
                                </div>
                                {showStats && (
                                  <div className="canvas-imputer__column-pill-stats">
                                    {hasMean && <span>Mean: {formatNumericStat(option.mean)}</span>}
                                    {hasMedian && <span>Median: {formatNumericStat(option.median)}</span>}
                                    {hasMode && <span>Mode: {formatModeStat(option.mode)}</span>}
                                  </div>
                                )}
                              </button>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="canvas-modal__note">
                          {hasCatalog
                            ? missingFilterActive
                              ? 'No columns meet the current missingness filter. Lower the threshold or add manual columns below.'
                              : 'No columns with missing values detected. Add columns manually or adjust upstream steps.'
                            : 'Column catalog unavailable; enter names manually below.'}
                        </p>
                      )}
                    </div>
                    <div
                      className={`canvas-imputer__selected${
                        strategy.columns.length ? '' : ' canvas-imputer__selected--empty'
                      }`}
                      aria-live="polite"
                    >
                      {strategy.columns.length ? (
                        strategy.columns.map((column) => (
                          <span key={column} className="canvas-imputer__selected-chip">
                            {column}
                          </span>
                        ))
                      ) : (
                        <span>No columns selected yet.</span>
                      )}
                    </div>
                    <FormField label="Manual columns (optional)">
                      <TextInput
                        value={columnsValue}
                        onChange={(event) => onColumnsChange(index, event.target.value)}
                        placeholder="Add custom columns, separated by commas"
                        aria-label="Manual column entry"
                      />
                    </FormField>
                    <p className="canvas-modal__note">
                      Tap columns above to toggle them. Missing % reflects the latest row-level preview, and manual entry lets you include custom fields.
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="canvas-modal__note">
          Define one or more strategies to fill missing values. Each strategy targets a set of columns and a method.
        </p>
      )}
    </section>
  );
};
