import React from 'react';
import type { SkewnessMethodStatus } from '../../../../api';
import {
  SKEWNESS_METHOD_ORDER,
  type SkewnessTableGroup,
  type SkewnessTableRow,
  type SkewnessTransformationMethod,
  type SkewnessViewMode,
} from './skewnessSettings';

type SkewnessInsightsSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  isFetchingSkewness: boolean;
  skewnessThreshold: number | null;
  skewnessError: string | null;
  skewnessViewMode: SkewnessViewMode;
  skewnessRecommendedCount: number;
  skewnessNumericCount: number;
  skewnessGroupByMethod: boolean;
  skewnessRows: SkewnessTableRow[];
  skewnessTableGroups: SkewnessTableGroup[];
  hasSkewnessAutoRecommendations: boolean;
  skewnessTransformationsCount: number;
  isFetchingRecommendations: boolean;
  getSkewnessMethodLabel: (method: SkewnessTransformationMethod) => string;
  getSkewnessMethodStatus: (
    column: string,
    method: SkewnessTransformationMethod,
  ) => SkewnessMethodStatus | { status: 'ready'; reason?: string };
  onApplyRecommendations: () => void;
  onViewModeChange: (mode: SkewnessViewMode) => void;
  onGroupByToggle: (checked: boolean) => void;
  onOverrideChange: (column: string, value: string) => void;
  onClearSelections: () => void;
};

export const SkewnessInsightsSection: React.FC<SkewnessInsightsSectionProps> = ({
  sourceId,
  hasReachableSource,
  isFetchingSkewness,
  skewnessThreshold,
  skewnessError,
  skewnessViewMode,
  skewnessRecommendedCount,
  skewnessNumericCount,
  skewnessGroupByMethod,
  skewnessRows,
  skewnessTableGroups,
  hasSkewnessAutoRecommendations,
  skewnessTransformationsCount,
  isFetchingRecommendations,
  getSkewnessMethodLabel,
  getSkewnessMethodStatus,
  onApplyRecommendations,
  onViewModeChange,
  onGroupByToggle,
  onOverrideChange,
  onClearSelections,
}) => {
  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Skewness insights</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onApplyRecommendations}
            disabled={isFetchingRecommendations || !hasSkewnessAutoRecommendations}
          >
            Apply recommendations
          </button>
        </div>
      </div>
      {skewnessThreshold !== null && (
        <p className="canvas-modal__note">
          Columns with |skewness| ≥ {skewnessThreshold.toFixed(2)} are surfaced below.
        </p>
      )}
      {isFetchingSkewness && skewnessRows.length === 0 && <p className="canvas-modal__note">Loading skewness diagnostics…</p>}
      {!isFetchingSkewness && skewnessError && (
        <p className="canvas-modal__note canvas-modal__note--error">{skewnessError}</p>
      )}
      {(!isFetchingSkewness || skewnessRows.length > 0) && !skewnessError && (
        <div className="canvas-skewness__body">
          <div className="canvas-skewness__toolbar">
            <div className="canvas-skewness__segmented" role="group" aria-label="Skewness view">
              <button
                type="button"
                className="canvas-skewness__segmented-button"
                data-active={skewnessViewMode === 'recommended'}
                onClick={() => onViewModeChange('recommended')}
                disabled={!skewnessRecommendedCount}
              >
                Recommended ({skewnessRecommendedCount})
              </button>
              <button
                type="button"
                className="canvas-skewness__segmented-button"
                data-active={skewnessViewMode === 'all'}
                onClick={() => onViewModeChange('all')}
                disabled={!skewnessNumericCount}
              >
                All numeric ({skewnessNumericCount})
              </button>
            </div>
            <label className="canvas-skewness__toggle">
              <input
                type="checkbox"
                checked={skewnessGroupByMethod}
                onChange={(event) => onGroupByToggle(event.target.checked)}
              />
              Group columns by recommended action
            </label>
          </div>
          {skewnessRows.length === 0 ? (
            <p className="canvas-modal__note">
              {skewnessViewMode === 'recommended'
                ? 'No columns met the skewness threshold for recommendations. Switch to “All numeric” to configure overrides manually.'
                : 'No numeric columns available for skewness review.'}
            </p>
          ) : (
            <div className="canvas-skewness__table-wrapper">
              {skewnessTableGroups.map((group) => (
                <div key={`skewness-group-${group.key}`} className="canvas-skewness__group">
                  {group.label && (
                    <div className="canvas-skewness__group-header">
                      <h4>{group.label}</h4>
                      <span>
                        {group.rows.length} column{group.rows.length === 1 ? '' : 's'}
                      </span>
                    </div>
                  )}
                  <div className="canvas-skewness__table-scroll">
                    <table className="canvas-skewness__table">
                      <thead>
                        <tr>
                          <th scope="col">Column</th>
                          <th scope="col">Suggested transform</th>
                          <th scope="col">Skewness level</th>
                          <th scope="col">Override</th>
                        </tr>
                      </thead>
                      <tbody>
                        {group.rows.map((row) => {
                          const parts: string[] = [];
                          if (row.directionLabel) {
                            parts.push(row.directionLabel);
                          }
                          if (row.skewnessValue !== null) {
                            parts.push(`(${row.skewnessValue.toFixed(2)})`);
                          }
                          if (row.magnitudeLabel) {
                            parts.push(`• ${row.magnitudeLabel}`);
                          }
                          const skewnessText = parts.length ? parts.join(' ') : '—';
                          const currentValue = row.selectedMethod ?? '';
                          const recommendedSet = new Set<SkewnessTransformationMethod>(row.recommendedMethods);

                          type SkewnessOption = {
                            value: string;
                            label: string;
                            status: SkewnessMethodStatus | null;
                            reason: string | null;
                          };

                          const optionEntries: SkewnessOption[] = [
                            {
                              value: '',
                              label: 'No transform',
                              status: null,
                              reason: null,
                            },
                          ];

                          row.recommendedMethods.forEach((method) => {
                            const status = getSkewnessMethodStatus(row.column, method);
                            optionEntries.push({
                              value: method,
                              label: `${getSkewnessMethodLabel(method)}${method === row.primaryMethod ? ' (recommended)' : ''}`,
                              status,
                              reason: status?.reason ?? null,
                            });
                          });

                          SKEWNESS_METHOD_ORDER.forEach((method) => {
                            if (recommendedSet.has(method)) {
                              return;
                            }
                            const status = getSkewnessMethodStatus(row.column, method);
                            optionEntries.push({
                              value: method,
                              label: getSkewnessMethodLabel(method),
                              status,
                              reason: status?.reason ?? null,
                            });
                          });

                          const currentStatus = row.selectedMethod
                            ? getSkewnessMethodStatus(row.column, row.selectedMethod)
                            : null;
                          const recommendedStatus = !row.selectedMethod && row.primaryMethod
                            ? getSkewnessMethodStatus(row.column, row.primaryMethod)
                            : null;

                          return (
                            <tr
                              key={`skewness-row-${row.column}`}
                              className={`canvas-skewness__row${row.selectedMethod ? ' canvas-skewness__row--active' : ''}`}
                            >
                              <th scope="row">
                                <div className="canvas-skewness__column-cell">
                                  <span className="canvas-skewness__column-name">{row.column}</span>
                                  {row.summary && (
                                    <span className="canvas-skewness__column-summary">{row.summary}</span>
                                  )}
                                </div>
                              </th>
                              <td>
                                {row.recommendedMethods.length ? (
                                  <span className={`canvas-skewness__chip${row.primaryMethod ? ' canvas-skewness__chip--recommended' : ''}`}>
                                    {row.primaryMethod
                                      ? getSkewnessMethodLabel(row.primaryMethod)
                                      : getSkewnessMethodLabel(row.recommendedMethods[0])}
                                  </span>
                                ) : (
                                  <span className="canvas-skewness__muted">No recommendation</span>
                                )}
                              </td>
                              <td>
                                <span className="canvas-skewness__metric">{skewnessText}</span>
                              </td>
                              <td>
                                <div className="canvas-skewness__override">
                                  <select
                                    value={currentValue}
                                    onChange={(event) => onOverrideChange(row.column, event.target.value)}
                                    disabled={isFetchingRecommendations || !hasReachableSource}
                                  >
                                    {optionEntries.map((option) => {
                                      const labelWithReason = option.reason
                                        ? `${option.label} (caution: ${option.reason})`
                                        : option.label;
                                      const statusToken = option.status?.status ?? 'ready';
                                      return (
                                        <option
                                          key={`skewness-option-${row.column}-${option.value || 'none'}`}
                                          value={option.value}
                                          data-status={statusToken}
                                        >
                                          {labelWithReason}
                                        </option>
                                      );
                                    })}
                                  </select>
                                  {currentStatus && currentStatus.status !== 'ready' && currentStatus.reason && (
                                    <span className="canvas-skewness__status canvas-skewness__status--error">
                                      Selected transform may be incompatible: {currentStatus.reason}
                                    </span>
                                  )}
                                  {!row.selectedMethod && recommendedStatus && recommendedStatus.status !== 'ready' && recommendedStatus.reason && (
                                    <span className="canvas-skewness__status">
                                      Recommended transform flagged: {recommendedStatus.reason}
                                    </span>
                                  )}
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="canvas-skewness__footer">
            <span>
              {skewnessTransformationsCount
                ? `${skewnessTransformationsCount} transformation${skewnessTransformationsCount === 1 ? '' : 's'} configured`
                : 'No transformations configured yet.'}
            </span>
            {skewnessTransformationsCount > 0 && (
              <button type="button" className="btn btn-outline-secondary" onClick={onClearSelections}>
                Clear selections
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
};
