import React, { useMemo, useState, useCallback, useEffect } from 'react';
import type {
  DateColumnOption,
  DateColumnSummary,
  DateFormatStrategyConfig,
  DateMode,
  DateModeOption,
  DateSampleMap,
} from './standardizeDateSettings';
import { DATE_MODE_OPTIONS, formatDateModeLabel } from './standardizeDateSettings';

export type StandardizeDatesSectionProps = {
  hasSource: boolean;
  hasReachableSource: boolean;
  strategies: DateFormatStrategyConfig[];
  columnSummary: DateColumnSummary;
  columnOptions: DateColumnOption[];
  sampleMap: DateSampleMap;
  collapsedStrategies: Set<number>;
  onToggleStrategy: (index: number) => void;
  onRemoveStrategy: (index: number) => void;
  onAddStrategy: () => void;
  onModeChange: (index: number, mode: DateMode) => void;
  onColumnToggle: (index: number, column: string) => void;
  onColumnsChange: (index: number, value: string) => void;
  onAutoDetectToggle: (index: number, enabled: boolean) => void;
  modeOptions?: DateModeOption[];
};

const formatSamplePreview = (samples: string[]): string | null => {
  if (!samples.length) {
    return null;
  }
  const preview = samples.slice(0, 3);
  const overflow = samples.length - preview.length;
  return overflow > 0 ? `${preview.join(', ')} …` : preview.join(', ');
};

export const StandardizeDatesSection: React.FC<StandardizeDatesSectionProps> = ({
  hasSource,
  hasReachableSource,
  strategies,
  columnSummary,
  columnOptions,
  sampleMap,
  collapsedStrategies,
  onToggleStrategy,
  onRemoveStrategy,
  onAddStrategy,
  onModeChange,
  onColumnToggle,
  onColumnsChange,
  onAutoDetectToggle,
  modeOptions = DATE_MODE_OPTIONS,
}) => {
  const hasStrategies = strategies.length > 0;
  const autoDetectActive = useMemo(
    () => strategies.some((strategy) => strategy.autoDetect),
    [strategies],
  );

  const [expandedColumnLists, setExpandedColumnLists] = useState<Set<number>>(() => new Set());

  const toggleAdditionalColumns = useCallback((index: number) => {
    setExpandedColumnLists((previous) => {
      const next = new Set(previous);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    setExpandedColumnLists((previous) => {
      const next = new Set<number>();
      previous.forEach((value) => {
        if (value < strategies.length) {
          next.add(value);
        }
      });
      return next.size === previous.size ? previous : next;
    });
  }, [strategies.length]);

  const columnUsage = useMemo(() => {
    const usage = new Map<string, number>();
    strategies.forEach((strategy) => {
      strategy.columns.forEach((column) => {
        const normalized = column.trim();
        if (!normalized) {
          return;
        }
        usage.set(normalized, (usage.get(normalized) ?? 0) + 1);
      });
    });
    return usage;
  }, [strategies]);

  const recommendedPreview = useMemo(() => {
    const preview = columnSummary.recommendedColumns.slice(0, 6);
    const remaining = Math.max(columnSummary.recommendedColumns.length - preview.length, 0);
    return { preview, remaining };
  }, [columnSummary.recommendedColumns]);

  const nonDateSummary = useMemo(() => {
    const preview = columnSummary.nonDateSelected.slice(0, 4);
    const remaining = Math.max(columnSummary.nonDateSelected.length - preview.length, 0);
    return { preview, remaining };
  }, [columnSummary.nonDateSelected]);

  const disconnectedWarning = !hasSource || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Date formatting strategies</h3>
        <div className="canvas-modal__section-actions">
          <button type="button" className="btn btn-outline-secondary" onClick={onAddStrategy}>
            Add strategy
          </button>
        </div>
      </div>

      {disconnectedWarning && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {hasSource
            ? 'Connect this step to an upstream output to explore detected date columns.'
            : 'Select a dataset to load column samples and suggestions.'}
        </p>
      )}

      {recommendedPreview.preview.length > 0 && (
        <p className="canvas-modal__note">
          Suggested columns: {recommendedPreview.preview.join(', ')}
          {recommendedPreview.remaining > 0 ? `, … (${recommendedPreview.remaining} more)` : ''}.
          {autoDetectActive
            ? ' Auto-detect strategies will automatically include remaining date-like columns.'
            : ''}
        </p>
      )}

      {columnSummary.sampleCandidates.length === 0 && !recommendedPreview.preview.length && (
        <p className="canvas-modal__note">
          No clear date candidates yet. Add columns manually or enable auto-detect to sweep for timestamps.
        </p>
      )}

      {nonDateSummary.preview.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {nonDateSummary.preview.join(', ')} {nonDateSummary.preview.length === 1 ? 'looks' : 'look'} non-date-like.
          {nonDateSummary.remaining > 0
            ? ` ${nonDateSummary.remaining} more selected column${nonDateSummary.remaining === 1 ? '' : 's'} may require double-checking.`
            : ' Double-check that these values represent consistent dates.'}
        </p>
      )}

      {autoDetectActive && (
        <p className="canvas-modal__note">
          Auto-detect gathers from {columnSummary.sampleCandidateCount || columnSummary.dateColumnCount} date-like columns that
          are not already assigned.
        </p>
      )}

      {hasStrategies ? (
        <div className="canvas-imputer__list">
          {strategies.map((strategy, index) => {
            const modeLabel = modeOptions.find((option) => option.value === strategy.mode)?.label ?? formatDateModeLabel(strategy.mode);
            const columnCount = strategy.columns.length;
            const columnSummaryLabel = columnCount
              ? `${columnCount} column${columnCount === 1 ? '' : 's'}`
              : strategy.autoDetect
                ? 'Auto-detecting columns'
                : 'No columns yet';
            const isCollapsed = collapsedStrategies.has(index);
            const columnsValue = strategy.columns.join(', ');
            const toggleId = `date-strategy-${index}-toggle`;
            const autoDetectId = `date-strategy-${index}-auto-detect`;

            const sortedOptions = columnOptions
              .map((option) => ({
                option,
                priority: option.isRecommended ? 0 : option.isSampleCandidate ? 1 : 2,
              }))
              .sort((a, b) => {
                if (a.priority !== b.priority) {
                  return a.priority - b.priority;
                }
                return a.option.name.localeCompare(b.option.name);
              })
              .map((entry) => entry.option);

            const primaryOptions = sortedOptions.filter((option) => {
              if (strategy.columns.includes(option.name)) {
                return true;
              }
              return option.isRecommended || option.isSampleCandidate;
            });
            const primarySet = new Set(primaryOptions.map((option) => option.name));
            const otherOptions = sortedOptions.filter((option) => !primarySet.has(option.name));
            const isOtherExpanded = expandedColumnLists.has(index);

            return (
              <div key={`date-strategy-${index}`} className="canvas-imputer__card">
                <div className="canvas-imputer__card-header">
                  <button
                    type="button"
                    id={toggleId}
                    className="canvas-imputer__card-toggle"
                    onClick={() => onToggleStrategy(index)}
                    aria-expanded={!isCollapsed}
                    aria-controls={`date-strategy-body-${index}`}
                  >
                    <span
                      className={`canvas-imputer__toggle-icon${isCollapsed ? '' : ' canvas-imputer__toggle-icon--open'}`}
                      aria-hidden="true"
                    />
                    <span className="canvas-imputer__card-text">
                      <span className="canvas-imputer__card-title">Strategy {index + 1}</span>
                      <span className="canvas-imputer__card-summary">
                        {modeLabel}
                        {strategy.autoDetect ? ' · auto-detect' : ''}
                        {columnSummaryLabel ? ` · ${columnSummaryLabel}` : ''}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="canvas-imputer__remove"
                    onClick={() => onRemoveStrategy(index)}
                    aria-label={`Remove date formatting strategy ${index + 1}`}
                  >
                    Remove
                  </button>
                </div>
                {!isCollapsed && (
                  <div className="canvas-imputer__card-body" id={`date-strategy-body-${index}`}>
                    <div className="canvas-imputer__row">
                      <label htmlFor={`date-strategy-${index}-mode`}>Format</label>
                      <select
                        id={`date-strategy-${index}-mode`}
                        value={strategy.mode}
                        onChange={(event) => onModeChange(index, event.target.value as DateMode)}
                      >
                        {modeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      <p className="canvas-modal__meta">
                        Example: {modeOptions.find((option) => option.value === strategy.mode)?.example ?? '2024-01-23'}
                      </p>
                    </div>

                    <div className="canvas-imputer__row">
                      <label htmlFor={autoDetectId}>Detection</label>
                      <div className="canvas-modal__boolean-control">
                        <input
                          id={autoDetectId}
                          type="checkbox"
                          checked={strategy.autoDetect}
                          onChange={(event) => onAutoDetectToggle(index, event.target.checked)}
                        />
                        <label htmlFor={autoDetectId}>Auto-detect additional date columns</label>
                      </div>
                      <p className="canvas-modal__note">
                        We will append detected date-like fields that are not assigned to other strategies.
                      </p>
                    </div>

                    <div className="canvas-imputer__row">
                      <label>Columns</label>
                      {sortedOptions.length ? (
                        <>
                          {primaryOptions.length ? (
                            <div
                              className="canvas-imputer__columns-list"
                              role="listbox"
                              aria-label="Detected or selected columns for date formatting"
                            >
                              {primaryOptions.map((option) => {
                                const isSelected = strategy.columns.includes(option.name);
                                const preview = formatSamplePreview(option.samples.length ? option.samples : sampleMap[option.name] ?? []);
                                const usageCount = columnUsage.get(option.name) ?? 0;
                                const assignedElsewhere = usageCount - (isSelected ? 1 : 0) > 0;
                                return (
                                  <button
                                    key={option.name}
                                    type="button"
                                    className={`canvas-imputer__column-pill${isSelected ? ' canvas-imputer__column-pill--selected' : ''}`}
                                    onClick={() => onColumnToggle(index, option.name)}
                                    aria-pressed={isSelected}
                                  >
                                    <span className="canvas-imputer__column-pill-name">{option.name}</span>
                                    <div className="canvas-imputer__column-pill-meta">
                                      {option.dtype && <span className="canvas-imputer__column-pill-dtype">{option.dtype}</span>}
                                      {option.isRecommended && <span className="canvas-modal__meta">Suggested</span>}
                                      {!option.isRecommended && option.isSampleCandidate && (
                                        <span className="canvas-modal__meta">Detected</span>
                                      )}
                                      {assignedElsewhere && <span className="canvas-modal__meta">Used in another strategy</span>}
                                    </div>
                                    {preview && <span className="canvas-modal__meta">e.g. {preview}</span>}
                                  </button>
                                );
                              })}
                            </div>
                          ) : (
                            <p className="canvas-modal__note">
                              No detected or selected columns yet. Expand other columns below to assign manually.
                            </p>
                          )}
                          {otherOptions.length > 0 && (
                            <div className="canvas-imputer__more-columns">
                              <button
                                type="button"
                                className="btn btn-outline-secondary"
                                onClick={() => toggleAdditionalColumns(index)}
                                aria-expanded={isOtherExpanded}
                                aria-controls={`date-strategy-${index}-other-columns`}
                              >
                                {isOtherExpanded
                                  ? 'Hide other columns'
                                  : `Show other columns (${otherOptions.length})`}
                              </button>
                              {isOtherExpanded && (
                                <div
                                  id={`date-strategy-${index}-other-columns`}
                                  className="canvas-imputer__columns-list"
                                  role="listbox"
                                  aria-label="Other available columns for date formatting"
                                >
                                  {otherOptions.map((option) => {
                                    const isSelected = strategy.columns.includes(option.name);
                                    const preview = formatSamplePreview(option.samples.length ? option.samples : sampleMap[option.name] ?? []);
                                    const usageCount = columnUsage.get(option.name) ?? 0;
                                    const assignedElsewhere = usageCount - (isSelected ? 1 : 0) > 0;
                                    return (
                                      <button
                                        key={option.name}
                                        type="button"
                                        className={`canvas-imputer__column-pill${isSelected ? ' canvas-imputer__column-pill--selected' : ''}`}
                                        onClick={() => onColumnToggle(index, option.name)}
                                        aria-pressed={isSelected}
                                      >
                                        <span className="canvas-imputer__column-pill-name">{option.name}</span>
                                        <div className="canvas-imputer__column-pill-meta">
                                          {option.dtype && <span className="canvas-imputer__column-pill-dtype">{option.dtype}</span>}
                                          {assignedElsewhere && <span className="canvas-modal__meta">Used in another strategy</span>}
                                        </div>
                                        {preview && <span className="canvas-modal__meta">e.g. {preview}</span>}
                                      </button>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="canvas-modal__note">
                          No catalogued columns yet. Add column names manually below or enable auto-detect to populate this list.
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
                        <span>No manual columns selected yet.</span>
                      )}
                    </div>

                    <div className="canvas-imputer__row">
                      <label htmlFor={`date-strategy-${index}-manual-columns`}>Manual columns (optional)</label>
                      <input
                        id={`date-strategy-${index}-manual-columns`}
                        type="text"
                        className="canvas-imputer__columns-input"
                        value={columnsValue}
                        onChange={(event) => onColumnsChange(index, event.target.value)}
                        placeholder="Add custom columns, separated by commas"
                      />
                    </div>

                    <p className="canvas-modal__note">
                      Tap suggestions to toggle columns. Manual entries let you set custom fields that are not listed above.
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="canvas-modal__note">
          Add a strategy to standardize one or more columns. You can combine manual selection with auto-detect for any
          remaining date-like fields.
        </p>
      )}
    </section>
  );
};
