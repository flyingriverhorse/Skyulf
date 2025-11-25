import React from 'react';
import { HistogramSparkline } from '../../utils/HistogramSparkline';
import { formatMetricValue, formatNumericStat } from '../../formatting';
import type { SkewnessColumnDistribution } from '../../../../api';
import type { SkewnessDistributionCard, SkewnessDistributionView } from '../../hooks/useSkewnessConfiguration';

type SkewnessDistributionSectionProps = {
  skewnessThreshold: number | null;
  isFetchingSkewness: boolean;
  skewnessError: string | null;
  skewnessDistributionCards: SkewnessDistributionCard[];
  skewnessDistributionView: SkewnessDistributionView;
  setSkewnessDistributionView: (view: SkewnessDistributionView) => void;
};

export const SkewnessDistributionSection: React.FC<SkewnessDistributionSectionProps> = ({
  skewnessThreshold,
  isFetchingSkewness,
  skewnessError,
  skewnessDistributionCards,
  skewnessDistributionView,
  setSkewnessDistributionView,
}) => {
  const renderDistributionSection = (
    distribution: SkewnessColumnDistribution,
    sectionLabel: string,
    methodLabel?: string | null,
  ) => {
    const validSamples = Math.max(0, distribution.sample_size || 0);
    const missingSamples = Math.max(0, distribution.missing_count || 0);
    return (
      <div className="canvas-skewness__distribution-block" key={sectionLabel}>
        <div className="canvas-skewness__distribution-block-title">
          <span>{sectionLabel}</span>
          {methodLabel ? (
            <span className="canvas-skewness__distribution-block-method">{methodLabel}</span>
          ) : null}
        </div>
        <HistogramSparkline
          counts={distribution.counts}
          binEdges={distribution.bin_edges}
          className="canvas-skewness__histogram"
        />
        <div className="canvas-skewness__distribution-stats">
          <div>
            <span className="canvas-skewness__stat-label">Min</span>
            <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.minimum)}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Median</span>
            <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.median)}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Max</span>
            <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.maximum)}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Mean</span>
            <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.mean)}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Std dev</span>
            <span className="canvas-skewness__stat-value">{formatNumericStat(distribution.stddev)}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Valid</span>
            <span className="canvas-skewness__stat-value">{validSamples.toLocaleString()}</span>
          </div>
          <div>
            <span className="canvas-skewness__stat-label">Missing</span>
            <span className="canvas-skewness__stat-value">{missingSamples.toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Skewness distributions</h3>
        <div className="canvas-modal__section-actions">
          <div className="canvas-skewness__segmented" role="group" aria-label="Distribution view">
            <button
              type="button"
              className="canvas-skewness__segmented-button"
              data-active={skewnessDistributionView === 'before'}
              onClick={() => setSkewnessDistributionView('before')}
              disabled={!skewnessDistributionCards.length}
            >
              Before
            </button>
            <button
              type="button"
              className="canvas-skewness__segmented-button"
              data-active={skewnessDistributionView === 'after'}
              onClick={() => setSkewnessDistributionView('after')}
              disabled={!skewnessDistributionCards.length}
            >
              After
            </button>
          </div>
        </div>
      </div>
      {skewnessThreshold !== null && (
        <p className="canvas-modal__note">
          Columns with |skewness| ≥ {skewnessThreshold.toFixed(2)} are visualized below.
        </p>
      )}
      {isFetchingSkewness && skewnessDistributionCards.length === 0 && <p className="canvas-modal__note">Loading skewness distributions…</p>}
      {!isFetchingSkewness && skewnessError && (
        <p className="canvas-modal__note canvas-modal__note--error">{skewnessError}</p>
      )}
      {(!isFetchingSkewness || skewnessDistributionCards.length > 0) && !skewnessError && (
        skewnessDistributionCards.length ? (
          <div className="canvas-skewness__distribution-grid">
            {skewnessDistributionCards.map((card) => {
              const afterDistribution = card.distributionAfter;
              const showBefore = skewnessDistributionView === 'before';
              const showAfter = skewnessDistributionView === 'after' && Boolean(afterDistribution);
              const showAfterPlaceholder = skewnessDistributionView === 'after' && !afterDistribution;

              const footnoteMessage = skewnessDistributionView === 'before'
                ? 'Showing the original distribution returned by the skewness analysis.'
                : afterDistribution
                  ? 'Showing the recomputed distribution after the applied transform.'
                  : 'A transformed distribution is not available yet for this column.';

              return (
                <article
                  key={`skewness-dist-${card.column}`}
                  className="canvas-skewness__distribution-card"
                  aria-label={`Distribution for ${card.column}`}
                >
                  <header className="canvas-skewness__distribution-header">
                    <div>
                      <h4>{card.column}</h4>
                      <div className="canvas-skewness__distribution-tags">
                        {card.directionLabel && (
                          <span className="canvas-skewness__chip canvas-skewness__chip--muted">{card.directionLabel}</span>
                        )}
                        {card.magnitudeLabel && (
                          <span className="canvas-skewness__chip canvas-skewness__chip--muted">{card.magnitudeLabel}</span>
                        )}
                        {card.recommendedLabel && (
                          <span className="canvas-skewness__chip canvas-skewness__chip--recommended">
                            Suggested: {card.recommendedLabel}
                          </span>
                        )}
                        {card.appliedLabel && (
                          <span className="canvas-skewness__chip canvas-skewness__chip--applied">
                            Applied: {card.appliedLabel}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="canvas-skewness__distribution-skew">
                      <span>Skewness</span>
                      <strong>{card.skewness !== null ? formatMetricValue(card.skewness, 2) : '—'}</strong>
                    </div>
                  </header>
                  {card.summary && (
                    <p className="canvas-skewness__distribution-summary">{card.summary}</p>
                  )}
                  <div className="canvas-skewness__distribution-comparison">
                    {showBefore
                      ? renderDistributionSection(card.distributionBefore, 'Before transform')
                      : null}
                    {showAfter && afterDistribution
                      ? renderDistributionSection(afterDistribution, 'After transform', card.appliedLabel)
                      : null}
                    {showAfterPlaceholder ? (
                      <div className="canvas-skewness__distribution-empty">
                        After transform data isn’t available yet. Apply a skewness method and rerun the node to
                        generate this view.
                      </div>
                    ) : null}
                  </div>
                  <footer className="canvas-skewness__distribution-footnote">
                    Histogram bins are based on the sampled rows returned by the skewness analysis. {footnoteMessage}
                  </footer>
                </article>
              );
            })}
          </div>
        ) : (
          <p className="canvas-modal__note">
            No columns met the skewness threshold for visualization. Refresh with a larger sample if needed.
          </p>
        )
      )}
    </section>
  );
};
