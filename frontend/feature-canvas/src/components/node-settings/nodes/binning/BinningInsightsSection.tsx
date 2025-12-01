import React, { useMemo } from 'react';
import { Trash2 } from 'lucide-react';
import type {
  BinningColumnRecommendation,
  BinningExcludedColumn,
} from '../../../../api';
import {
  BINNING_STRATEGY_LABELS,
  BINNING_STRATEGY_OPTIONS,
  type BinningColumnOverride,
  type BinningStrategy,
  type KBinsEncode,
  type KBinsStrategy,
  type NormalizedBinningConfig,
} from './binningSettings';

const formatStrategyLabel = (strategy: string) =>
  BINNING_STRATEGY_LABELS[strategy as BinningStrategy] ?? strategy.replace(/_/g, ' ');

const formatConfidenceLabel = (value: string | null | undefined) => {
  switch (value) {
    case 'high':
      return 'High confidence';
    case 'medium':
      return 'Medium confidence';
    case 'low':
      return 'Low confidence';
    default:
      return 'Confidence unknown';
  }
};

const renderStatsSummary = (
  stats: BinningColumnRecommendation['stats'] | null | undefined,
  formatMetricValue: (value: number) => string,
  formatNumericStat: (value: number) => string,
) => {
  if (!stats || typeof stats !== 'object') {
    return 'No distribution summary';
  }
  const parts: string[] = [];
  if (typeof stats.valid_count === 'number' && Number.isFinite(stats.valid_count)) {
    parts.push(`${formatMetricValue(stats.valid_count)} valid rows`);
  }
  if (typeof stats.distinct_count === 'number' && Number.isFinite(stats.distinct_count)) {
    parts.push(`${formatMetricValue(stats.distinct_count)} distinct values`);
  }
  const hasMin = typeof stats.minimum === 'number' && Number.isFinite(stats.minimum);
  const hasMax = typeof stats.maximum === 'number' && Number.isFinite(stats.maximum);
  if (hasMin || hasMax) {
    const minDisplay = hasMin ? formatNumericStat(stats.minimum as number) : '—';
    const maxDisplay = hasMax ? formatNumericStat(stats.maximum as number) : '—';
    parts.push(`Range ${minDisplay} – ${maxDisplay}`);
  }
  if (typeof stats.skewness === 'number' && Number.isFinite(stats.skewness)) {
    parts.push(`Skew ${formatNumericStat(stats.skewness)}`);
  }
  return parts.length ? parts.join(' · ') : 'No distribution summary';
};

type BinningInsightsSectionProps = {
  hasSource: boolean;
  hasReachableSource: boolean;
  isFetching: boolean;
  error: string | null;
  relativeGeneratedAt: string | null;
  sampleSize: number | null;
  binningConfig: NormalizedBinningConfig;
  binningDefaultLabel: string;
  binningOverrideCount: number;
  binningOverrideSummary: string | null;
  recommendations: BinningColumnRecommendation[];
  excludedColumns: BinningExcludedColumn[];
  numericColumnCount: number;
  canApplyAllNumeric: boolean;
  onApplyAllNumeric?: () => void;
  onApplyStrategies?: () => void;
  onClearOverrides: () => void;
  customEdgeDrafts: Record<string, string>;
  customLabelDrafts: Record<string, string>;
  onOverrideStrategyChange: (
    column: string,
    value: BinningStrategy | '__default__',
    options?: { recommendedBins?: number | null },
  ) => void;
  onOverrideNumberChange: (
    column: string,
    field: 'equal_width_bins' | 'equal_frequency_bins' | 'kbins_n_bins',
    value: string,
  ) => void;
  onOverrideKbinsEncodeChange: (column: string, value: KBinsEncode | null) => void;
  onOverrideKbinsStrategyChange: (column: string, value: KBinsStrategy | null) => void;
  onOverrideClear: (column: string) => void;
  onCustomBinsChange: (column: string, rawValue: string) => void;
  onCustomLabelsChange: (column: string, rawValue: string) => void;
  onClearCustomColumn: (column: string) => void;
  formatMetricValue: (value: number) => string;
  formatNumericStat: (value: number) => string;
};

export const BinningInsightsSection: React.FC<BinningInsightsSectionProps> = ({
  hasSource,
  hasReachableSource,
  isFetching,
  error,
  relativeGeneratedAt,
  sampleSize,
  binningConfig,
  binningDefaultLabel,
  binningOverrideCount,
  binningOverrideSummary,
  recommendations,
  excludedColumns,
  numericColumnCount,
  canApplyAllNumeric,
  onApplyAllNumeric,
  onApplyStrategies,
  onClearOverrides,
  customEdgeDrafts,
  customLabelDrafts,
  onOverrideStrategyChange,
  onOverrideNumberChange,
  onOverrideKbinsEncodeChange,
  onOverrideKbinsStrategyChange,
  onOverrideClear,
  onCustomBinsChange,
  onCustomLabelsChange,
  onClearCustomColumn,
  formatMetricValue,
  formatNumericStat,
}) => {
  const summaryLabel = useMemo(() => {
    if (!hasSource) {
      return 'Select a dataset to surface diagnostics.';
    }
    if (!hasReachableSource) {
      return 'Connect this step to an upstream output to surface diagnostics.';
    }
    if (isFetching) {
      return 'Diagnostics are loading…';
    }
    if (error) {
      return 'Diagnostics are unavailable at the moment.';
    }
    if (numericColumnCount > 0) {
      return `Detected ${numericColumnCount} numeric column${numericColumnCount === 1 ? '' : 's'} in the current sample.`;
    }
    if (excludedColumns.length > 0) {
      return `Diagnostics skipped ${excludedColumns.length} column${excludedColumns.length === 1 ? '' : 's'} (see details).`;
    }
    return 'No additional diagnostics were produced for the current sample.';
  }, [error, excludedColumns.length, hasReachableSource, hasSource, isFetching, numericColumnCount]);

  const excludedPreview = useMemo(() => {
    const preview = excludedColumns.slice(0, 6);
    const remaining = Math.max(0, excludedColumns.length - preview.length);
    return {
      preview,
      remaining,
    };
  }, [excludedColumns]);

  const recommendationMap = useMemo(() => {
    const map = new Map<string, BinningColumnRecommendation>();
    recommendations.forEach((entry) => {
      const column = typeof entry?.column === 'string' ? entry.column.trim() : '';
      if (column) {
        map.set(column, entry);
      }
    });
    return map;
  }, [recommendations]);

  const allColumns = useMemo(() => {
    const set = new Set<string>();
    binningConfig.columns.forEach((column) => {
      const normalized = String(column ?? '').trim();
      if (normalized) {
        set.add(normalized);
      }
    });
    Object.keys(binningConfig.columnStrategies).forEach((column) => {
      const normalized = String(column ?? '').trim();
      if (normalized) {
        set.add(normalized);
      }
    });
    recommendations.forEach((entry) => {
      const column = typeof entry?.column === 'string' ? entry.column.trim() : '';
      if (column) {
        set.add(column);
      }
    });
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [binningConfig.columnStrategies, binningConfig.columns, recommendations]);

  const resolveSelectValue = (override: BinningColumnOverride | undefined, defaultStrategy: BinningStrategy) => {
    if (!override || Object.keys(override).length === 0) {
      return '__default__';
    }
    if (override.strategy) {
      return override.strategy;
    }
    if (override.customBins?.length) {
      return 'custom';
    }
    if (override.kbinsNBins || override.kbinsEncode || override.kbinsStrategy) {
      return 'kbins';
    }
    if (override.equalFrequencyBins) {
      return 'equal_frequency';
    }
    if (override.equalWidthBins) {
      return 'equal_width';
    }
    return defaultStrategy;
  };

  const hasValidContext = hasSource && hasReachableSource;

  const buildOverrideDetails = (override: BinningColumnOverride | undefined) => {
    if (!override) {
      return [] as string[];
    }
    const parts: string[] = [];
    if (typeof override.equalWidthBins === 'number') {
      parts.push(`${override.equalWidthBins} bins`);
    }
    if (typeof override.equalFrequencyBins === 'number') {
      parts.push(`${override.equalFrequencyBins} bins`);
    }
    if (typeof override.kbinsNBins === 'number') {
      parts.push(`${override.kbinsNBins} bins`);
    }
    if (override.kbinsStrategy) {
      parts.push(override.kbinsStrategy);
    }
    if (override.kbinsEncode) {
      parts.push(override.kbinsEncode);
    }
    if (override.customBins?.length) {
      parts.push('Custom bins');
    }
    if (override.customLabels?.length) {
      parts.push('Custom labels');
    }
    return parts;
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Binning diagnostics</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onClearOverrides}
            disabled={!binningOverrideCount}
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <Trash2 size={14} />
            Clear
          </button>
          {onApplyStrategies && (
            <button
              type="button"
              className="btn btn-primary"
              onClick={onApplyStrategies}
              disabled={!recommendations.length}
            >
              Quick apply
            </button>
          )}
        </div>
      </div>
      <p className="canvas-modal__note">{summaryLabel}</p>
      <p className="canvas-modal__note">
        Default strategy <strong>{binningDefaultLabel}</strong>
        {binningOverrideCount > 0
          ? ` · Overrides: ${binningOverrideCount}${
              binningOverrideSummary ? ` (${binningOverrideSummary})` : ''
            }`
          : ' · No column overrides yet'}
      </p>
      <div className="canvas-modal__collapsible">
        {!hasSource && (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to surface binning insights and column diagnostics.
          </p>
        )}
        {hasSource && !hasReachableSource && (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this step to an upstream output to analyse numeric feature distributions.
          </p>
        )}
        {hasValidContext && (
          <>
            {relativeGeneratedAt && (
              <p className="canvas-modal__note">
                Insights generated {relativeGeneratedAt}
                {sampleSize !== null ? ` on ${formatMetricValue(sampleSize)} row${sampleSize === 1 ? '' : 's'}.` : '.'}
              </p>
            )}
            {onApplyAllNumeric && numericColumnCount > 0 && (
              <div className="canvas-modal__section-actions">
                <button
                  type="button"
                  className="btn btn-outline-secondary"
                  onClick={onApplyAllNumeric}
                  disabled={!canApplyAllNumeric}
                >
                  {`Add all numeric columns (${numericColumnCount})`}
                </button>
              </div>
            )}
            {isFetching && <p className="canvas-modal__note">Loading binning diagnostics…</p>}
            {error && !isFetching && (
              <p className="canvas-modal__note canvas-modal__note--error">{error}</p>
            )}
            {!error && (
              <>
                {allColumns.length > 0 && (
                  <div className="canvas-cast__body">
                    <div className="canvas-cast__table-wrapper">
                      <table className="canvas-cast__table">
                        <thead>
                          <tr>
                            <th scope="col">Column</th>
                            <th scope="col">Suggested strategy</th>
                            <th scope="col">Configured strategy</th>
                            <th scope="col">Distribution highlights</th>
                          </tr>
                        </thead>
                        <tbody>
                          {allColumns.map((column) => {
                              const recommendation = recommendationMap.get(column) ?? null;
                              const override = binningConfig.columnStrategies[column];
                              const selectValue = resolveSelectValue(override, binningConfig.strategy);
                              const effectiveStrategy =
                                selectValue === '__default__'
                                  ? binningConfig.strategy
                                  : (selectValue as BinningStrategy);
                              const recommendedStrategy = recommendation?.recommended_strategy ?? null;
                              const needsAttention = Boolean(
                                recommendedStrategy && recommendedStrategy !== effectiveStrategy,
                              );
                              const isOverrideApplied = Boolean(override && Object.keys(override).length);
                              const isSelected = binningConfig.columns.includes(column);
                              const recommendedBinsRaw = recommendation?.recommended_bins;
                              const recommendedBinsNumber =
                                typeof recommendedBinsRaw === 'number'
                                  ? recommendedBinsRaw
                                  : recommendedBinsRaw !== null && recommendedBinsRaw !== undefined
                                  ? Number(recommendedBinsRaw)
                                  : null;
                              const recommendedBins =
                                typeof recommendedBinsNumber === 'number' &&
                                Number.isFinite(recommendedBinsNumber) &&
                                recommendedBinsNumber > 0
                                  ? recommendedBinsNumber
                                  : null;
                              const equalWidthValue = override?.equalWidthBins ?? binningConfig.equalWidthBins;
                              const equalFrequencyValue =
                                override?.equalFrequencyBins ?? binningConfig.equalFrequencyBins;
                              const kbinsNBinsValue = override?.kbinsNBins ?? binningConfig.kbinsNBins;
                              const kbinsEncodeValue = override?.kbinsEncode ?? '__default__';
                              const kbinsStrategyValue = override?.kbinsStrategy ?? '__default__';
                              const rowClasses = ['canvas-cast__row'];
                              if (needsAttention) {
                                rowClasses.push('canvas-cast__row--attention');
                              }
                              if (isOverrideApplied) {
                                rowClasses.push('canvas-cast__row--override');
                              }
                              if (!isSelected) {
                                rowClasses.push('canvas-cast__row--muted');
                              }
                              const statsSummary = renderStatsSummary(
                                recommendation?.stats,
                                formatMetricValue,
                                formatNumericStat,
                              );
                              const reasons = recommendation?.reasons?.filter((reason) => reason && reason.trim()) ?? [];
                              const overrideDetails = (() => {
                                const details = buildOverrideDetails(override);
                                const hasCustomBins = Array.isArray(binningConfig.customBins[column])
                                  ? binningConfig.customBins[column].length >= 2
                                  : false;
                                const hasCustomLabels = Array.isArray(binningConfig.customLabels[column])
                                  ? binningConfig.customLabels[column].length > 0
                                  : false;
                                const merged = [...details];
                                if (hasCustomBins && !merged.includes('Custom bins')) {
                                  merged.push('Custom bins');
                                }
                                if (hasCustomLabels && !merged.includes('Custom labels')) {
                                  merged.push('Custom labels');
                                }
                                return merged;
                              })();
                              const helperText =
                                selectValue === '__default__'
                                  ? `Inheriting ${binningDefaultLabel}`
                                  : `Override → ${formatStrategyLabel(effectiveStrategy)}${
                                      overrideDetails.length ? ` (${overrideDetails.join(', ')})` : ''
                                    }`;
                              const dtype = recommendation?.dtype ?? null;
                              const confidenceLabel = recommendation
                                ? formatConfidenceLabel(recommendation.confidence)
                                : null;
                              const customEdgesDraft = Object.prototype.hasOwnProperty.call(customEdgeDrafts, column)
                                ? customEdgeDrafts[column]
                                : binningConfig.customBins[column]?.join(', ') ?? '';
                              const customLabelsDraft = Object.prototype.hasOwnProperty.call(customLabelDrafts, column)
                                ? customLabelDrafts[column]
                                : binningConfig.customLabels[column]?.join(', ') ?? '';
                              const hasCustomValues = Boolean(
                                (binningConfig.customBins[column]?.length ?? 0) >= 2 ||
                                  (binningConfig.customLabels[column]?.length ?? 0) > 0,
                              );
                              return (
                                <tr key={`binning-insight-${column}`} className={rowClasses.join(' ')}>
                                  <th scope="row">
                                    <div className="canvas-cast__column-cell">
                                      <span className="canvas-cast__column-name">{column}</span>
                                      <div className="canvas-cast__column-meta">
                                        {dtype && <span className="canvas-cast__muted">dtype: {dtype}</span>}
                                        {!isSelected && (
                                          <span className="canvas-cast__muted">Not currently in node</span>
                                        )}
                                        {isOverrideApplied && (
                                          <span className="canvas-cast__badge canvas-cast__badge--override">Override</span>
                                        )}
                                        {needsAttention && (
                                          <span className="canvas-cast__badge canvas-cast__badge--attention">
                                            Needs review
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  </th>
                                  <td>
                                    <div className="canvas-cast__recommendation">
                                      {recommendation ? (
                                        <>
                                          <span className="canvas-cast__chip canvas-cast__chip--applied">
                                            {formatStrategyLabel(recommendation.recommended_strategy)}
                                          </span>
                                          <span className="canvas-cast__recommendation-note">
                                            {recommendedBins !== null
                                              ? `${recommendedBins} bin${recommendedBins === 1 ? '' : 's'}`
                                              : 'Bins not provided'}
                                            {confidenceLabel ? ` · ${confidenceLabel}` : ''}
                                          </span>
                                          {reasons.length > 0 && (
                                            <span className="canvas-cast__recommendation-note">
                                              {reasons.join('; ')}
                                            </span>
                                          )}
                                        </>
                                      ) : (
                                        <span className="canvas-cast__muted">No recommendation</span>
                                      )}
                                    </div>
                                  </td>
                                  <td>
                                    <div className="canvas-cast__target">
                                      <select
                                        value={selectValue}
                                        onChange={(event) =>
                                          onOverrideStrategyChange(
                                            column,
                                            event.target.value as BinningStrategy | '__default__',
                                            { recommendedBins },
                                          )
                                        }
                                      >
                                        <option value="__default__">Inherit default ({binningDefaultLabel})</option>
                                        {BINNING_STRATEGY_OPTIONS.map((option) => (
                                          <option key={`${column}-${option.value}`} value={option.value}>
                                            {option.label}
                                          </option>
                                        ))}
                                      </select>
                                      <span className="canvas-cast__target-note">{helperText}</span>
                                      {selectValue === 'equal_width' && (
                                        <div className="canvas-cast__target-note">
                                          <label>
                                            Integer bins
                                            <input
                                              type="number"
                                              min={2}
                                              max={200}
                                              className="canvas-modal__input"
                                              value={equalWidthValue ?? ''}
                                              onChange={(event) =>
                                                onOverrideNumberChange(column, 'equal_width_bins', event.target.value)
                                              }
                                            />
                                          </label>
                                        </div>
                                      )}
                                      {selectValue === 'equal_frequency' && (
                                        <div className="canvas-cast__target-note">
                                          <label>
                                            Integer bins
                                            <input
                                              type="number"
                                              min={2}
                                              max={200}
                                              className="canvas-modal__input"
                                              value={equalFrequencyValue ?? ''}
                                              onChange={(event) =>
                                                onOverrideNumberChange(
                                                  column,
                                                  'equal_frequency_bins',
                                                  event.target.value,
                                                )
                                              }
                                            />
                                          </label>
                                        </div>
                                      )}
                                      {selectValue === 'kbins' && (
                                        <div className="canvas-cast__target-note">
                                          <label>
                                            Max bins
                                            <input
                                              type="number"
                                              min={2}
                                              max={200}
                                              className="canvas-modal__input"
                                              value={kbinsNBinsValue ?? ''}
                                              onChange={(event) =>
                                                onOverrideNumberChange(column, 'kbins_n_bins', event.target.value)
                                              }
                                            />
                                          </label>
                                          <label>
                                            Encode
                                            <select
                                              value={kbinsEncodeValue}
                                              onChange={(event) =>
                                                onOverrideKbinsEncodeChange(
                                                  column,
                                                  event.target.value === '__default__'
                                                    ? null
                                                    : (event.target.value as KBinsEncode),
                                                )
                                              }
                                            >
                                              <option value="__default__">Default ({binningConfig.kbinsEncode})</option>
                                              <option value="ordinal">Ordinal</option>
                                              <option value="onehot">One-hot (sparse)</option>
                                              <option value="onehot-dense">One-hot (dense)</option>
                                            </select>
                                          </label>
                                          <label>
                                            Strategy
                                            <select
                                              value={kbinsStrategyValue}
                                              onChange={(event) =>
                                                onOverrideKbinsStrategyChange(
                                                  column,
                                                  event.target.value === '__default__'
                                                    ? null
                                                    : (event.target.value as KBinsStrategy),
                                                )
                                              }
                                            >
                                              <option value="__default__">Default ({binningConfig.kbinsStrategy})</option>
                                              <option value="uniform">Uniform</option>
                                              <option value="quantile">Quantile</option>
                                              <option value="kmeans">K-means</option>
                                            </select>
                                          </label>
                                        </div>
                                      )}
                                      {selectValue === 'custom' && (
                                        <div className="canvas-cast__target-note">
                                          <label>
                                            Custom edges
                                            <input
                                              type="text"
                                              className="canvas-modal__input"
                                              value={customEdgesDraft}
                                              placeholder="e.g. 0, 50, 100"
                                              onChange={(event) => onCustomBinsChange(column, event.target.value)}
                                            />
                                          </label>
                                          <label>
                                            Optional labels
                                            <input
                                              type="text"
                                              className="canvas-modal__input"
                                              value={customLabelsDraft}
                                              placeholder="e.g. Low, Medium, High"
                                              onChange={(event) => onCustomLabelsChange(column, event.target.value)}
                                            />
                                          </label>
                                          {hasCustomValues && (
                                            <button
                                              type="button"
                                              className="btn btn-link"
                                              onClick={() => onClearCustomColumn(column)}
                                            >
                                              Clear custom edges
                                            </button>
                                          )}
                                        </div>
                                      )}
                                      {isOverrideApplied && (
                                        <button
                                          type="button"
                                          className="btn btn-link"
                                          onClick={() => onOverrideClear(column)}
                                        >
                                          Clear
                                        </button>
                                      )}
                                    </div>
                                  </td>
                                  <td>
                                    <span>{statsSummary}</span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                )}
                {!isFetching && allColumns.length === 0 && (
                  <p className="canvas-modal__note">
                    No columns met the sampling requirements for binning recommendations.
                  </p>
                )}
                {excludedColumns.length > 0 && (
                  <p className="canvas-modal__note">
                    Skipped {excludedColumns.length} column{excludedColumns.length === 1 ? '' : 's'} due to non-numeric values or
                    limited variance
                    {excludedPreview.preview.length > 0 && (
                      <>
                        :{' '}
                        {excludedPreview.preview
                          .map((entry) => `${entry.column}${entry.reason ? ` (${entry.reason})` : ''}`)
                          .join(', ')}
                        {excludedPreview.remaining > 0 ? `, … (+${excludedPreview.remaining} more)` : ''}
                      </>
                    )}
                    .
                  </p>
                )}
              </>
            )}
            </>
          )}
        </div>
    </section>
  );
};
