import React from 'react';
import type { ScalingMethodDetail, ScalingMethodName } from '../../../../api';
import type {
  NormalizedScalingConfig,
  ScalingMethodOption,
  ScalingRecommendationRow,
} from './scalingSettings';

type ScalingInsightsSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  isFetchingScaling: boolean;
  scalingConfig: NormalizedScalingConfig;
  scalingMethodOptions: ScalingMethodOption[];
  scalingMethodDetailMap: Map<ScalingMethodName, ScalingMethodDetail>;
  scalingDefaultDetail: ScalingMethodDetail | null;
  scalingAutoDetectEnabled: boolean;
  scalingSelectedCount: number;
  scalingDefaultLabel: string;
  scalingOverrideCount: number;
  scalingOverrideExampleSummary: string | null;
  scalingRecommendationRows: ScalingRecommendationRow[];
  scalingHasRecommendations: boolean;
  scalingStatusMessage: string | null;
  scalingError: string | null;
  scalingSampleSize: number | null;
  relativeScalingGeneratedAt: string | null;
  formatMetricValue: (value: number) => string;
  formatNumericStat: (value: number) => string;
  formatMissingPercentage: (value: number | null) => string;
  onApplyAllRecommendations: () => void;
  onClearOverrides: () => void;
  onDefaultMethodChange: (method: ScalingMethodName) => void;
  onAutoDetectToggle: (checked: boolean) => void;
  onOverrideSelect: (column: string, value: string) => void;
};

export const ScalingInsightsSection: React.FC<ScalingInsightsSectionProps> = ({
  sourceId,
  hasReachableSource,
  isFetchingScaling,
  scalingConfig,
  scalingMethodOptions,
  scalingMethodDetailMap,
  scalingDefaultDetail,
  scalingAutoDetectEnabled,
  scalingSelectedCount,
  scalingDefaultLabel,
  scalingOverrideCount,
  scalingOverrideExampleSummary,
  scalingRecommendationRows,
  scalingHasRecommendations,
  scalingStatusMessage,
  scalingError,
  scalingSampleSize,
  relativeScalingGeneratedAt,
  formatMetricValue,
  formatNumericStat,
  formatMissingPercentage,
  onApplyAllRecommendations,
  onClearOverrides,
  onDefaultMethodChange,
  onAutoDetectToggle,
  onOverrideSelect,
}) => {
  const defaultDetail = scalingDefaultDetail ?? scalingMethodDetailMap.get(scalingConfig.defaultMethod) ?? null;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Scaling insights</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onApplyAllRecommendations}
            disabled={!scalingHasRecommendations}
          >
            Apply recommendations
          </button>
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onClearOverrides}
            disabled={!scalingOverrideCount}
          >
            Clear overrides
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Automatically select scaling strategies for numeric features to keep model inputs in comparable ranges. Edits here refresh the preview immediately so you can inspect the effect, but nothing is committed to your pipeline until you click <strong>Save changes</strong> at the bottom of this modal.
      </p>
      {!sourceId && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to surface scaling insights and column recommendations.
        </p>
      )}
      {sourceId && !hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this step to an upstream output to analyse numeric feature distributions.
        </p>
      )}
      {relativeScalingGeneratedAt && (
        <p className="canvas-modal__note">
          Insights generated {relativeScalingGeneratedAt}
          {scalingSampleSize !== null
            ? ` on ${formatMetricValue(scalingSampleSize)} row${scalingSampleSize === 1 ? '' : 's'}.`
            : '.'}
        </p>
      )}
      <div className="canvas-modal__parameter-grid">
        <div className="canvas-modal__parameter-field">
          <div className="canvas-modal__parameter-label">
            <span>Default scaling method</span>
          </div>
          <div className="canvas-scaling__method-toggle" role="group" aria-label="Default scaling method">
            {scalingMethodOptions.map((option) => {
              const methodDetail = scalingMethodDetailMap.get(option.value) ?? null;
              const isActive = option.value === scalingConfig.defaultMethod;
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
          {defaultDetail?.description && (
            <p className="canvas-modal__parameter-description">{defaultDetail.description}</p>
          )}
        </div>
        <div className="canvas-modal__parameter-field">
          <div className="canvas-modal__parameter-label">
            <span>Auto-detect numeric columns</span>
          </div>
          <label className="canvas-modal__boolean-control">
            <input
              type="checkbox"
              checked={scalingAutoDetectEnabled}
              onChange={(event) => onAutoDetectToggle(event.target.checked)}
            />
            <span>{scalingAutoDetectEnabled ? 'Enabled' : 'Disabled'}</span>
          </label>
          <p className="canvas-modal__parameter-description">
            When enabled, recommended numeric features join this node automatically after each refresh.
          </p>
        </div>
      </div>
      <p className="canvas-modal__note">
        Tracking <strong>{scalingSelectedCount}</strong> column{scalingSelectedCount === 1 ? '' : 's'} in this node · Default method{' '}
        <strong>{scalingDefaultLabel}</strong>{' '}
        {scalingAutoDetectEnabled ? '· Auto-detect on' : '· Auto-detect off'}
        {scalingOverrideCount > 0
          ? ` · Overrides: ${scalingOverrideCount}${
              scalingOverrideExampleSummary ? ` (${scalingOverrideExampleSummary})` : ''
            }`
          : ' · No column overrides yet'}
      </p>
      {isFetchingScaling && <p className="canvas-modal__note">Loading scaling diagnostics…</p>}
      {!isFetchingScaling && scalingError && sourceId && hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--error">{scalingError}</p>
      )}
      {!isFetchingScaling && !scalingError && (
        <>
          {scalingStatusMessage && <p className="canvas-modal__note">{scalingStatusMessage}</p>}
          {scalingRecommendationRows.length > 0 && (
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
                    {scalingRecommendationRows.map((row) => {
                      const rowNeedsAttention = !row.isSkipped && row.recommendedMethod !== row.currentMethod;
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
                          ? 'Skipped (column will not be scaled)'
                          : row.isSelected
                            ? `Inheriting ${scalingDefaultLabel}`
                            : 'Pending inclusion';
                      const stats = row.stats;
                      const statsParts: string[] = [];
                      if (typeof stats.valid_count === 'number' && Number.isFinite(stats.valid_count)) {
                        statsParts.push(`${formatMetricValue(stats.valid_count)} valid rows`);
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
                      if (typeof stats.iqr === 'number' && Number.isFinite(stats.iqr)) {
                        statsParts.push(`IQR ${formatNumericStat(stats.iqr)}`);
                      }
                      const hasMin = typeof stats.minimum === 'number' && Number.isFinite(stats.minimum);
                      const hasMax = typeof stats.maximum === 'number' && Number.isFinite(stats.maximum);
                      if (hasMin || hasMax) {
                        const minDisplay = hasMin ? formatNumericStat(stats.minimum as number) : '—';
                        const maxDisplay = hasMax ? formatNumericStat(stats.maximum as number) : '—';
                        statsParts.push(`Range ${minDisplay} – ${maxDisplay}`);
                      }
                      if (typeof stats.skewness === 'number' && Number.isFinite(stats.skewness)) {
                        statsParts.push(`Skew ${formatNumericStat(stats.skewness)}`);
                      }
                      if (
                        typeof stats.outlier_ratio === 'number' &&
                        Number.isFinite(stats.outlier_ratio) &&
                        stats.outlier_ratio > 0
                      ) {
                        statsParts.push(`${formatMissingPercentage(stats.outlier_ratio * 100)} outliers`);
                      }
                      const statsSummary = statsParts.length ? statsParts.join(' · ') : null;
                      return (
                        <tr key={`scaling-row-${row.column}`} className={rowClasses.join(' ')}>
                          <th scope="row">
                            <div className="canvas-cast__column-cell">
                              <span className="canvas-cast__column-name">{row.column}</span>
                              <div className="canvas-cast__column-meta">
                                {row.dtype && <span className="canvas-cast__muted">dtype: {row.dtype}</span>}
                                {row.isExcluded && (
                                  <span className="canvas-cast__muted">Excluded from scaling (non-numeric)</span>
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
                              </div>
                            </div>
                          </th>
                          <td>
                            <div className="canvas-cast__recommendation">
                              <span
                                className={`canvas-cast__chip${
                                  rowNeedsAttention
                                    ? ' canvas-cast__chip--attention'
                                    : ' canvas-cast__chip--applied'
                                }`}
                              >
                                {row.recommendedLabel}
                              </span>
                              <span className="canvas-cast__recommendation-note">{row.confidenceLabel}</span>
                              {row.reasons.length > 0 && (
                                <span className="canvas-cast__recommendation-note">{row.reasons.join('; ')}</span>
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
                              <select value={selectValue} onChange={(event) => onOverrideSelect(row.column, event.target.value)}>
                                <option value="__default__">Inherit default ({scalingDefaultLabel})</option>
                                <option value="__skip__">Skip scaling for this column</option>
                                {scalingMethodOptions.map((option) => (
                                  <option key={`${row.column}-${option.value}`} value={option.value}>
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                              <span className="canvas-cast__target-note">{overrideHelper}</span>
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
