import React, { useMemo, useEffect } from 'react';
import {
  BINNED_SAMPLE_PRESETS,
  type BinnedDistributionCard,
  type BinnedSamplePresetValue,
} from './binningSettings';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import { extractPendingConfigurationDetails, type PendingConfigurationDetail } from '../../utils/pendingConfiguration';

type BinnedDistributionSectionProps = {
  cards: BinnedDistributionCard[];
  selectedPreset: BinnedSamplePresetValue;
  onSelectPreset: (value: BinnedSamplePresetValue) => void;
  isFetching: boolean;
  sampleSize: number | null;
  relativeGeneratedAt: string | null;
  error: string | null;
  onRefresh: () => void;
  canRefresh: boolean;
  previewState?: PreviewState;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
};

export const BinnedDistributionSection: React.FC<BinnedDistributionSectionProps> = ({
  cards,
  selectedPreset,
  onSelectPreset,
  isFetching,
  sampleSize,
  relativeGeneratedAt,
  error,
  onRefresh,
  canRefresh,
  previewState,
  onPendingConfigurationWarning,
}) => {
  useEffect(() => {
    if (previewState?.data?.signals?.full_execution && onPendingConfigurationWarning) {
      const details = extractPendingConfigurationDetails(previewState.data.signals.full_execution);
      if (details.length > 0) {
        onPendingConfigurationWarning(details);
      }
    }
  }, [previewState, onPendingConfigurationWarning]);

  const activePreset = useMemo(() => {
    return BINNED_SAMPLE_PRESETS.find((preset) => preset.value === selectedPreset) ?? BINNED_SAMPLE_PRESETS[0];
  }, [selectedPreset]);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Binned column distributions</h3>
        <div className="canvas-modal__section-actions">
          <div className="canvas-sample__group">
            <span className="canvas-sample__label">Sample</span>
            <div className="canvas-skewness__segmented" role="group" aria-label="Binned sampling presets">
              {BINNED_SAMPLE_PRESETS.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  className="canvas-skewness__segmented-button"
                  data-active={preset.value === selectedPreset}
                  onClick={() => onSelectPreset(preset.value)}
                  disabled={isFetching}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onRefresh}
            disabled={isFetching || !canRefresh}
          >
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>
      {sampleSize !== null && (
        <p className="canvas-modal__note">
          Sampling preset: <strong>{activePreset.label}</strong>.{' '}
          {activePreset.value === 'all'
            ? `Summaries computed from the full dataset (${sampleSize.toLocaleString()} rows).`
            : `Summaries computed from ${sampleSize.toLocaleString()} sampled rows.`}
        </p>
      )}
      {relativeGeneratedAt && (
        <p className="canvas-modal__note">Updated {relativeGeneratedAt}</p>
      )}
      {isFetching && <p className="canvas-modal__note">Loading binned distributions…</p>}
      {!isFetching && error && <p className="canvas-modal__note canvas-modal__note--error">{error}</p>}
      {!isFetching && !error && (
        cards.length ? (
          <div className="canvas-binning__grid">
            {cards.map((card) => {
              const topPercentageLabel = card.topPercentage !== null ? `${card.topPercentage.toFixed(1)}%` : null;
              const summaryValue = card.topCount !== null
                ? `${card.topCount.toLocaleString()}${topPercentageLabel ? ` · ${topPercentageLabel}` : ''}`
                : topPercentageLabel ?? '—';
              return (
                <article key={`binned-card-${card.column}`} className="canvas-binning__card">
                  <header className="canvas-binning__header">
                    <div>
                      <h4>{card.column}</h4>
                      {card.sourceColumn && card.sourceColumn !== card.column && (
                        <p className="canvas-binning__subtitle">
                          Derived from <strong>{card.sourceColumn}</strong>
                        </p>
                      )}
                    </div>
                    <div className="canvas-binning__summary" aria-label="Top bin summary">
                      <span className="canvas-binning__summary-label">Top bin</span>
                      <span className="canvas-binning__summary-value">{card.topLabel ?? '—'}</span>
                      <span className="canvas-binning__summary-meta">{summaryValue}</span>
                    </div>
                  </header>
                  <div className="canvas-binning__meta">
                    <div>
                      <span className="canvas-binning__meta-label">Sample size</span>
                      <span className="canvas-binning__meta-value">{card.totalRows.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="canvas-binning__meta-label">Distinct bins</span>
                      <span className="canvas-binning__meta-value">{card.distinctBins.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="canvas-binning__meta-label">Missing rows</span>
                      <span className="canvas-binning__meta-value">{card.missingRows.toLocaleString()}</span>
                    </div>
                  </div>
                  <div className="canvas-binning__bins">
                    {card.bins.map((bin) => {
                      const labelClass = bin.isMissing
                        ? 'canvas-binning__bin-label canvas-binning__bin-label--missing'
                        : 'canvas-binning__bin-label';
                      const barWidth = Math.min(100, Math.max(0, bin.percentage));
                      return (
                        <div className="canvas-binning__bin-row" key={`${card.column}-${bin.label}`}>
                          <div className={labelClass}>{bin.label}</div>
                          <div className="canvas-binning__bin-bar" aria-hidden="true">
                            <div className="canvas-binning__bin-bar-fill" style={{ width: `${barWidth}%` }} />
                          </div>
                          <div className="canvas-binning__bin-stats">
                            <span>{bin.count.toLocaleString()}</span>
                            <span>{bin.percentage.toFixed(1)}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {card.hasMoreBins && (
                    <p className="canvas-binning__note">
                      Showing top {card.bins.length} of {card.totalBinCount} bins.
                    </p>
                  )}
                </article>
              );
            })}
          </div>
        ) : (
          <p className="canvas-modal__note">
            No binned columns detected upstream. Connect this node after a binning step to view category distributions.
          </p>
        )
      )}
    </section>
  );
};
