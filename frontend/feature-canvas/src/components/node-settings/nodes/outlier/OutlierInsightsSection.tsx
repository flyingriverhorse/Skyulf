import React, { useEffect } from 'react';
import { Trash2 } from 'lucide-react';
import { CheckboxInput, NumberInput } from '../../ui/FormFields';
import type {
  OutlierMethodDetail,
  OutlierMethodName,
  OutlierNodeSignal,
} from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';
import {
  OUTLIER_METHOD_FALLBACK_LABELS,
  OUTLIER_METHOD_ORDER,
  OUTLIER_METHOD_DEFAULT_PARAMETERS,
  OUTLIER_PARAMETER_KEYS,
  type NormalizedOutlierConfig,
  type OutlierMethodOption,
  type OutlierRecommendationRow,
} from './outlierSettings';

const formatParameterLabel = (methodDetail: OutlierMethodDetail | null, parameter: string): string => {
  const help = methodDetail?.parameter_help?.[parameter];
  if (help) {
    return help;
  }
  const fallback = parameter.replace(/_/g, ' ').trim();
  return fallback.charAt(0).toUpperCase() + fallback.slice(1);
};

const resolvePreviewSummary = (signal: OutlierNodeSignal | null | undefined): string | null => {
  if (!signal) {
    return null;
  }
  const parts: string[] = [];
  if (signal.removed_rows && signal.removed_rows > 0) {
    parts.push(`Removed ${signal.removed_rows} row${signal.removed_rows === 1 ? '' : 's'}`);
  }
  if (Array.isArray(signal.clipped_columns) && signal.clipped_columns.length) {
    const unique = Array.from(new Set(signal.clipped_columns));
    parts.push(`Winsorized ${unique.length} column${unique.length === 1 ? '' : 's'}`);
  }
  if (!parts.length) {
    parts.push('No rows removed or clipped during the last preview');
  }
  if (Array.isArray(signal.skipped_columns) && signal.skipped_columns.length) {
    const skipped = signal.skipped_columns.join(', ');
    parts.push(`Skipped columns: ${skipped}`);
  }
  return parts.join(' · ');
};

const resolveParameterValue = (
  method: OutlierMethodName,
  parameter: string,
  columnParameters: Record<string, Record<string, number>>,
  column: string,
  methodParameters: Record<OutlierMethodName, Record<string, number>>,
): { value: number | '';
  fallback: number | null } => {
  const columnParameterMap = columnParameters[column] ?? {};
  if (Object.prototype.hasOwnProperty.call(columnParameterMap, parameter)) {
    const numeric = columnParameterMap[parameter];
    if (typeof numeric === 'number' && Number.isFinite(numeric)) {
      return { value: numeric, fallback: null };
    }
  }
  const globalParameters = methodParameters[method] ?? {};
  const globalValue = globalParameters[parameter];
  if (typeof globalValue === 'number' && Number.isFinite(globalValue)) {
    return { value: '', fallback: globalValue };
  }
  const defaultValue = OUTLIER_METHOD_DEFAULT_PARAMETERS[method]?.[parameter] ?? null;
  return { value: '', fallback: defaultValue ?? null };
};

const resolvePlaceholder = (fallback: number | null): string => {
  if (typeof fallback === 'number' && Number.isFinite(fallback)) {
    return String(fallback);
  }
  return '';
};

const PARAMETER_SUFFIX: Record<string, string> = {
  lower_percentile: '%',
  upper_percentile: '%',
};

type OutlierInsightsSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  isFetchingOutliers: boolean;
  outlierConfig: NormalizedOutlierConfig;
  outlierMethodOptions: OutlierMethodOption[];
  outlierMethodDetailMap: Map<OutlierMethodName, OutlierMethodDetail>;
  outlierDefaultDetail: OutlierMethodDetail | null;
  outlierAutoDetectEnabled: boolean;
  outlierSelectedCount: number;
  outlierDefaultLabel: string;
  outlierOverrideCount: number;
  outlierParameterOverrideCount: number;
  outlierOverrideExampleSummary: string | null;
  outlierHasOverrides: boolean;
  outlierRecommendationRows: OutlierRecommendationRow[];
  outlierHasRecommendations: boolean;
  outlierStatusMessage: string | null;
  outlierError: string | null;
  outlierSampleSize: number | null;
  relativeOutlierGeneratedAt: string | null;
  outlierPreviewSignal: OutlierNodeSignal | null;
  formatMetricValue: (value: number) => string;
  formatNumericStat: (value: number) => string;
  onApplyAllRecommendations: () => void;
  onClearOverrides: () => void;
  onDefaultMethodChange: (method: OutlierMethodName) => void;
  onAutoDetectToggle: (checked: boolean) => void;
  onOverrideSelect: (column: string, value: string) => void;
  onMethodParameterChange: (method: OutlierMethodName, parameter: string, value: number | null) => void;
  onColumnParameterChange: (column: string, parameter: string, value: number | null) => void;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
  onPendingConfigurationCleared?: () => void;
};

export const OutlierInsightsSection: React.FC<OutlierInsightsSectionProps> = ({
  sourceId,
  hasReachableSource,
  isFetchingOutliers,
  outlierConfig,
  outlierMethodOptions,
  outlierMethodDetailMap,
  outlierDefaultDetail,
  outlierAutoDetectEnabled,
  outlierSelectedCount,
  outlierDefaultLabel,
  outlierOverrideCount,
  outlierParameterOverrideCount,
  outlierOverrideExampleSummary,
  outlierHasOverrides,
  outlierRecommendationRows,
  outlierHasRecommendations,
  outlierStatusMessage,
  outlierError,
  outlierSampleSize,
  relativeOutlierGeneratedAt,
  outlierPreviewSignal,
  formatMetricValue,
  formatNumericStat,
  onApplyAllRecommendations,
  onClearOverrides,
  onDefaultMethodChange,
  onAutoDetectToggle,
  onOverrideSelect,
  onMethodParameterChange,
  onColumnParameterChange,
  previewState,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      
      let relevantDetails = details;

      // If auto-detect is enabled, suppress "columns not configured" warnings
      if (outlierAutoDetectEnabled) {
        relevantDetails = relevantDetails.filter((d) => !d.label.toLowerCase().includes('columns'));
      }

      if (relevantDetails.length > 0) {
        onPendingConfigurationWarning?.(relevantDetails);
      } else {
        onPendingConfigurationCleared?.();
      }
    }
  }, [previewState, onPendingConfigurationWarning, onPendingConfigurationCleared, outlierAutoDetectEnabled]);

  const previewSummary = resolvePreviewSummary(outlierPreviewSignal);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Outlier insights</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onApplyAllRecommendations}
            disabled={!outlierHasRecommendations}
          >
            Apply recommendations
          </button>
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onClearOverrides}
            disabled={!outlierHasOverrides}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <Trash2 size={14} />
            Clear overrides
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Diagnose numeric outliers and configure how to handle them across the dataset. Preview updates immediately, letting you validate the impact before saving this node.
      </p>
      {!sourceId && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to surface outlier diagnostics.
        </p>
      )}
      {sourceId && !hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this step to an upstream output to analyse numeric distributions.
        </p>
      )}
      {relativeOutlierGeneratedAt && (
        <p className="canvas-modal__note">
          Insights generated {relativeOutlierGeneratedAt}
          {outlierSampleSize !== null
            ? ` on ${formatMetricValue(outlierSampleSize)} row${outlierSampleSize === 1 ? '' : 's'}.`
            : '.'}
        </p>
      )}
      {!relativeOutlierGeneratedAt && outlierSampleSize !== null && (
        <p className="canvas-modal__note">
          Insights generated on {formatMetricValue(outlierSampleSize)} row{outlierSampleSize === 1 ? '' : 's'}.
        </p>
      )}
      {previewSummary && <p className="canvas-modal__note">Last preview: {previewSummary}</p>}
      <div className="canvas-modal__parameter-grid">
        <div className="canvas-modal__parameter-field">
          <div className="canvas-modal__parameter-label">
            <span>Default outlier method</span>
          </div>
          <div className="canvas-scaling__method-toggle" role="group" aria-label="Default outlier method">
            {outlierMethodOptions.map((option) => {
              const methodDetail = outlierMethodDetailMap.get(option.value) ?? null;
              const isActive = option.value === outlierConfig.defaultMethod;
              return (
                <button
                  key={option.value}
                  type="button"
                  className={`canvas-scaling__method-button${isActive ? ' canvas-scaling__method-button--active' : ''}`}
                  aria-pressed={isActive}
                  onClick={() => onDefaultMethodChange(option.value)}
                >
                  <span className="canvas-scaling__method-label">{option.label}</span>
                  {methodDetail?.description && (
                    <span className="canvas-scaling__method-description">{methodDetail.description}</span>
                  )}
                </button>
              );
            })}
          </div>
          {outlierDefaultDetail?.description && (
            <p className="canvas-modal__parameter-description">{outlierDefaultDetail.description}</p>
          )}
        </div>
        <CheckboxInput
          fieldLabel="Auto-detect numeric columns"
          label={outlierAutoDetectEnabled ? 'Enabled' : 'Disabled'}
          checked={outlierAutoDetectEnabled}
          onChange={(event) => onAutoDetectToggle(event.target.checked)}
          description="Auto-detection tracks numeric features so newly profiled columns join the node automatically."
        />
      </div>

      {(() => {
        const method = outlierConfig.defaultMethod;
        const parameters = OUTLIER_PARAMETER_KEYS[method] ?? [];

        if (!parameters.length) {
          return null;
        }

        const detail = outlierMethodDetailMap.get(method) ?? null;

        return (
          <section className="canvas-modal__parameter-field">
            <div className="canvas-modal__parameter-label">
              <span>Global method parameters</span>
            </div>
            <p className="canvas-modal__parameter-description canvas-outlier__method-note">
              Tune default thresholds for the selected strategy. Column overrides can refine these further.
            </p>
            <div className="canvas-outlier__method-grid">
              <article className="canvas-outlier__method-card">
                <div className="canvas-outlier__parameter-list">
                  {parameters.map((parameter) => {
                    const value = outlierConfig.methodParameters[method]?.[parameter] ?? null;
                    const suffix = PARAMETER_SUFFIX[parameter] ?? '';
                    return (
                      <label key={`${method}-${parameter}`} className="canvas-outlier__method-parameter">
                        <span className="canvas-outlier__parameter-label">
                          {formatParameterLabel(detail, parameter)}
                        </span>
                        <div className="canvas-outlier__method-input">
                          <NumberInput
                            className="canvas-outlier__method-input-field"
                            value={typeof value === 'number' && Number.isFinite(value) ? value : ''}
                            placeholder={resolvePlaceholder(
                              OUTLIER_METHOD_DEFAULT_PARAMETERS[method]?.[parameter] ?? null,
                            )}
                            onChange={(event) => {
                              const raw = event.target.value;
                              if (raw === '') {
                                onMethodParameterChange(method, parameter, null);
                              } else {
                                const parsed = Number(raw);
                                onMethodParameterChange(
                                  method,
                                  parameter,
                                  Number.isFinite(parsed) ? parsed : null,
                                );
                              }
                            }}
                          />
                          {suffix && <span className="canvas-outlier__parameter-suffix">{suffix}</span>}
                        </div>
                      </label>
                    );
                  })}
                </div>
              </article>
            </div>
          </section>
        );
      })()}

      <p className="canvas-modal__note">
        Tracking <strong>{outlierSelectedCount}</strong> column{outlierSelectedCount === 1 ? '' : 's'} · Default method{' '}
        <strong>{outlierDefaultLabel}</strong>{' '}
        {outlierAutoDetectEnabled ? '· Auto-detect on' : '· Auto-detect off'}
        {(() => {
          const totalOverrides = outlierOverrideCount + outlierParameterOverrideCount;
          if (!totalOverrides) {
            return ' · No overrides yet';
          }
          const details: string[] = [];
          if (outlierOverrideCount > 0) {
            details.push(
              outlierOverrideExampleSummary
                ? `${outlierOverrideCount} method override${outlierOverrideCount === 1 ? '' : 's'} (${outlierOverrideExampleSummary})`
                : `${outlierOverrideCount} method override${outlierOverrideCount === 1 ? '' : 's'}`,
            );
          }
          if (outlierParameterOverrideCount > 0) {
            details.push(
              `${outlierParameterOverrideCount} parameter override${outlierParameterOverrideCount === 1 ? '' : 's'}`,
            );
          }
          return ` · Overrides: ${details.join('; ')}`;
        })()}
      </p>

      {isFetchingOutliers && <p className="canvas-modal__note">Loading outlier diagnostics…</p>}
      {!isFetchingOutliers && outlierError && sourceId && hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--error">{outlierError}</p>
      )}
      {!isFetchingOutliers && !outlierError && (
        <>
          {outlierStatusMessage && <p className="canvas-modal__note">{outlierStatusMessage}</p>}
          {outlierRecommendationRows.length > 0 && (
            <div className="canvas-cast__body">
              <div className="canvas-cast__table-wrapper">
                <table className="canvas-cast__table canvas-scaling__table">
                  <thead>
                    <tr>
                      <th scope="col">Column</th>
                      <th scope="col">Recommendation</th>
                      <th scope="col">Distribution</th>
                      <th scope="col">Configured method</th>
                    </tr>
                  </thead>
                  <tbody>
                    {outlierRecommendationRows.map((row) => {
                      const rowNeedsAttention =
                        !row.isSkipped && !row.isExcluded && row.recommendedMethod && row.recommendedMethod !== row.currentMethod;
                      const rowClasses = ['canvas-cast__row'];
                      if (rowNeedsAttention) {
                        rowClasses.push('canvas-cast__row--attention');
                      }
                      if (row.isOverrideApplied) {
                        rowClasses.push('canvas-cast__row--override');
                      }
                      if (row.isSkipped) {
                        rowClasses.push('canvas-cast__row--muted');
                      }

                      const selectValue = row.isSkipped
                        ? '__skip__'
                        : row.isOverrideApplied
                          ? row.currentMethod
                          : '__default__';
                      const overrideHelper = row.isOverrideApplied
                        ? `Override -> ${row.currentMethodLabel}`
                        : row.isSkipped
                          ? 'Skipped (column will not be altered)'
                          : row.isSelected
                            ? `Inheriting ${outlierDefaultLabel}`
                            : 'Pending inclusion';

                      const stats = row.stats;
                      const statsParts: string[] = [];
                      if (typeof stats.valid_count === 'number' && Number.isFinite(stats.valid_count)) {
                        statsParts.push(`${formatMetricValue(stats.valid_count)} valid rows`);
                      }
                      const hasMin = typeof stats.minimum === 'number' && Number.isFinite(stats.minimum);
                      const hasMax = typeof stats.maximum === 'number' && Number.isFinite(stats.maximum);
                      if (hasMin || hasMax) {
                        const minDisplay = hasMin ? formatNumericStat(stats.minimum as number) : '—';
                        const maxDisplay = hasMax ? formatNumericStat(stats.maximum as number) : '—';
                        statsParts.push(`Range ${minDisplay} – ${maxDisplay}`);
                      }
                      if (typeof stats.mean === 'number' && Number.isFinite(stats.mean)) {
                        statsParts.push(`Mean ${formatNumericStat(stats.mean)}`);
                      }
                      if (typeof stats.median === 'number' && Number.isFinite(stats.median)) {
                        statsParts.push(`Median ${formatNumericStat(stats.median)}`);
                      }
                      if (typeof stats.stddev === 'number' && Number.isFinite(stats.stddev)) {
                        statsParts.push(`σ ${formatNumericStat(stats.stddev)}`);
                      }
                      const statsSummary = statsParts.length ? statsParts.join(' · ') : null;

                      const recommendationSummary = (() => {
                        if (!row.recommendedMethod || !row.recommendedLabel) {
                          return row.recommendedReason || 'No outlier risk detected';
                        }
                        return row.recommendedReason
                          ? `${row.recommendedLabel} – ${row.recommendedReason}`
                          : row.recommendedLabel;
                      })();

                      const columnParameters = outlierConfig.columnParameters;
                      const methodParameters = outlierConfig.methodParameters;
                      const parameterKeys = OUTLIER_PARAMETER_KEYS[row.currentMethod] ?? [];

                      return (
                        <tr key={`outlier-row-${row.column}`} className={rowClasses.join(' ')}>
                          <th scope="row">
                            <div className="canvas-cast__column-cell">
                              <span className="canvas-cast__column-name">{row.column}</span>
                              <div className="canvas-cast__column-meta">
                                {row.dtype && <span className="canvas-cast__muted">dtype: {row.dtype}</span>}
                                {row.isExcluded && (
                                  <span className="canvas-cast__muted">Excluded from outlier evaluation (non-numeric)</span>
                                )}
                                {row.isSelected && !row.isExcluded && (
                                  <span className="canvas-cast__badge canvas-cast__badge--selected">In node</span>
                                )}
                                {row.isSkipped && !row.isExcluded && (
                                  <span className="canvas-cast__badge canvas-cast__badge--skipped">Skipped</span>
                                )}
                                {row.isOverrideApplied && (
                                  <span className="canvas-cast__badge canvas-cast__badge--override">Override</span>
                                )}
                                {row.hasMissing && (
                                  <span className="canvas-cast__muted">Contains missing values</span>
                                )}
                              </div>
                            </div>
                          </th>
                          <td>
                            <div className="canvas-cast__recommendation">
                              {row.recommendedLabel ? (
                                <span
                                  className={`canvas-cast__chip${
                                    rowNeedsAttention
                                      ? ' canvas-cast__chip--attention'
                                      : ' canvas-cast__chip--applied'
                                  }`}
                                >
                                  {row.recommendedLabel}
                                </span>
                              ) : (
                                <span className="canvas-cast__chip">No action needed</span>
                              )}
                              {recommendationSummary && (
                                <span className="canvas-cast__recommendation-note">{recommendationSummary}</span>
                              )}
                            </div>
                          </td>
                          <td>
                            {statsSummary ? (
                              <span>{statsSummary}</span>
                            ) : (
                              <span className="canvas-cast__muted">No distribution summary</span>
                            )}
                          </td>
                          <td>
                            <div className="canvas-cast__target">
                              <select
                                value={selectValue}
                                onChange={(event) => onOverrideSelect(row.column, event.target.value)}
                                disabled={row.isExcluded}
                              >
                                <option value="__default__">Inherit default ({outlierDefaultLabel})</option>
                                <option value="__skip__">Skip outlier handling</option>
                                {outlierMethodOptions.map((option) => (
                                  <option key={`${row.column}-${option.value}`} value={option.value}>
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                              <span className="canvas-cast__target-note">{overrideHelper}</span>
                              {parameterKeys.length > 0 && !row.isExcluded && !row.isSkipped && (
                                <div className="canvas-outlier__parameters">
                                  {parameterKeys.map((parameterKey) => {
                                    const { value, fallback } = resolveParameterValue(
                                      row.currentMethod,
                                      parameterKey,
                                      columnParameters,
                                      row.column,
                                      methodParameters,
                                    );
                                    const suffix = PARAMETER_SUFFIX[parameterKey] ?? '';
                                    return (
                                      <label key={`${row.column}-${row.currentMethod}-${parameterKey}`}>
                                        <span>{parameterKey.replace(/_/g, ' ')}</span>
                                        <NumberInput
                                          value={value}
                                          placeholder={resolvePlaceholder(fallback)}
                                          onChange={(event) => {
                                            const raw = event.target.value;
                                            if (raw === '') {
                                              onColumnParameterChange(row.column, parameterKey, null);
                                            } else {
                                              const parsed = Number(raw);
                                              onColumnParameterChange(
                                                row.column,
                                                parameterKey,
                                                Number.isFinite(parsed) ? parsed : null,
                                              );
                                            }
                                          }}
                                        />
                                        {suffix && <span className="canvas-outlier__parameter-suffix">{suffix}</span>}
                                      </label>
                                    );
                                  })}
                                </div>
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
          )}
        </>
      )}
    </section>
  );
};
