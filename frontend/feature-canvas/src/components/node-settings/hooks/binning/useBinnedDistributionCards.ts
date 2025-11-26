import { useMemo } from 'react';
import type { BinnedDistributionResponse, BinnedColumnDistribution, BinnedColumnBin } from '../../../../api';
import type { BinnedDistributionBin, BinnedDistributionCard } from '../../nodes/binning/binningSettings';
import { type CatalogFlagMap } from '../core/useCatalogFlags';

type UseBinnedDistributionCardsArgs = {
  catalogFlags: CatalogFlagMap;
  binnedDistributionData: BinnedDistributionResponse | null;
};

export const useBinnedDistributionCards = ({
  catalogFlags,
  binnedDistributionData,
}: UseBinnedDistributionCardsArgs): BinnedDistributionCard[] => {
  const { isBinnedDistributionNode } = catalogFlags;
  return useMemo<BinnedDistributionCard[]>(() => {
    if (!isBinnedDistributionNode) {
      return [];
    }

    const rawColumns = Array.isArray(binnedDistributionData?.columns) ? binnedDistributionData?.columns : [];

    const cards = rawColumns
      .map((entry: BinnedColumnDistribution) => {
        if (!entry || !entry.column) {
          return null;
        }

        const totalRowsNumeric = Number(entry.total_rows);
        const totalRows = Number.isFinite(totalRowsNumeric) ? Math.max(0, Math.round(totalRowsNumeric)) : 0;
        if (totalRows <= 0) {
          return null;
        }

        const rawBins = Array.isArray(entry.bins) ? entry.bins : [];
        const sanitizedBins = rawBins
          .map((bin: BinnedColumnBin): BinnedDistributionBin | null => {
            if (!bin) {
              return null;
            }

            const labelRaw = typeof bin.label === 'string' ? bin.label.trim() : String(bin.label ?? '');
            const label = labelRaw || (bin.is_missing ? 'Missing' : 'Unlabeled bin');

            const numericCount = Number(bin.count);
            if (!Number.isFinite(numericCount)) {
              return null;
            }

            const safeCount = Math.max(0, Math.round(numericCount));

            const numericPercentage = Number(bin.percentage);
            const clampedPercentage = Number.isFinite(numericPercentage)
              ? Math.min(100, Math.max(0, numericPercentage))
              : 0;
            const roundedPercentage = Number(clampedPercentage.toFixed(2));

            return {
              label,
              count: safeCount,
              percentage: roundedPercentage,
              isMissing: Boolean(bin.is_missing),
            };
          })
          .filter((bin): bin is BinnedDistributionBin => Boolean(bin));

        if (!sanitizedBins.length) {
          return null;
        }

        sanitizedBins.sort((a: BinnedDistributionBin, b: BinnedDistributionBin) => {
          if (a.isMissing !== b.isMissing) {
            return a.isMissing ? 1 : -1;
          }
          if (a.count !== b.count) {
            return b.count - a.count;
          }
          return a.label.localeCompare(b.label);
        });

        const totalBinCount = sanitizedBins.length;
        const MAX_BINS = 12;
        const hasMoreBins = totalBinCount > MAX_BINS;
        const bins = hasMoreBins ? sanitizedBins.slice(0, MAX_BINS) : sanitizedBins;

        const missingRowsNumeric = Number(entry.missing_rows);
        const missingRows = Number.isFinite(missingRowsNumeric) ? Math.max(0, Math.round(missingRowsNumeric)) : 0;

        const distinctBinsNumeric = Number(entry.distinct_bins);
        const distinctBins = Number.isFinite(distinctBinsNumeric)
          ? Math.max(0, Math.round(distinctBinsNumeric))
          : sanitizedBins.length;

        const rawTopLabel = entry.top_label;
        let topLabel = typeof rawTopLabel === 'string' ? rawTopLabel.trim() || null : rawTopLabel ?? null;
        const topCountNumeric = Number(entry.top_count);
        let topCount = Number.isFinite(topCountNumeric) ? Math.max(0, Math.round(topCountNumeric)) : null;
        const topPercentageNumeric = Number(entry.top_percentage);
        let topPercentage = Number.isFinite(topPercentageNumeric)
          ? Number(Math.min(100, Math.max(0, topPercentageNumeric)).toFixed(2))
          : null;

        if ((topLabel === null || topCount === null || topPercentage === null) && bins.length) {
          topLabel = bins[0].label;
          topCount = bins[0].count;
          topPercentage = bins[0].percentage;
        }

        const sourceColumn = typeof entry.source_column === 'string' ? entry.source_column.trim() || null : entry.source_column ?? null;

        return {
          column: entry.column,
          sourceColumn,
          totalRows,
          missingRows,
          distinctBins,
          topLabel,
          topCount,
          topPercentage,
          bins,
          hasMoreBins,
          totalBinCount,
        } as BinnedDistributionCard;
      })
      .filter((card): card is BinnedDistributionCard => Boolean(card));

    cards.sort((a: BinnedDistributionCard, b: BinnedDistributionCard) => {
      const diff = (b.topCount ?? 0) - (a.topCount ?? 0);
      if (diff !== 0) {
        return diff;
      }
      return a.column.localeCompare(b.column);
    });

    return cards;
  }, [binnedDistributionData, isBinnedDistributionNode]);
};
