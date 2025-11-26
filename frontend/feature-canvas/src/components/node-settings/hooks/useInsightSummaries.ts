import { useMemo } from 'react';
import { formatRelativeTime } from '../utils/formatters';

interface GeneratedDataLike {
  generated_at?: string | null;
  sample_size?: string | number | null;
}

interface UseInsightSummariesArgs {
  recommendationsGeneratedAt: string | null | undefined;
  scalingData: GeneratedDataLike | null | undefined;
  binningData: GeneratedDataLike | null | undefined;
  binnedDistributionData: GeneratedDataLike | null | undefined;
}

interface UseInsightSummariesResult {
  relativeGeneratedAt: string | null;
  relativeScalingGeneratedAt: string | null;
  relativeBinningGeneratedAt: string | null;
  relativeBinnedGeneratedAt: string | null;
  scalingSampleSize: number | null;
  binningSampleSize: number | null;
  binnedSampleSize: number | null;
}

const normalizeSampleSize = (value: string | number | null | undefined): number | null => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return Math.max(0, Math.round(numeric));
};

export const useInsightSummaries = ({
  recommendationsGeneratedAt,
  scalingData,
  binningData,
  binnedDistributionData,
}: UseInsightSummariesArgs): UseInsightSummariesResult => {
  const relativeGeneratedAt = useMemo(
    () => formatRelativeTime(recommendationsGeneratedAt ?? null),
    [recommendationsGeneratedAt],
  );
  const relativeScalingGeneratedAt = useMemo(
    () => formatRelativeTime(scalingData?.generated_at ?? null),
    [scalingData?.generated_at],
  );
  const relativeBinningGeneratedAt = useMemo(
    () => formatRelativeTime(binningData?.generated_at ?? null),
    [binningData?.generated_at],
  );
  const relativeBinnedGeneratedAt = useMemo(
    () => formatRelativeTime(binnedDistributionData?.generated_at ?? null),
    [binnedDistributionData?.generated_at],
  );

  const scalingSampleSize = useMemo(
    () => normalizeSampleSize(scalingData?.sample_size ?? null),
    [scalingData?.sample_size],
  );
  const binningSampleSize = useMemo(
    () => normalizeSampleSize(binningData?.sample_size ?? null),
    [binningData?.sample_size],
  );
  const binnedSampleSize = useMemo(
    () => normalizeSampleSize(binnedDistributionData?.sample_size ?? null),
    [binnedDistributionData?.sample_size],
  );

  return {
    relativeGeneratedAt,
    relativeScalingGeneratedAt,
    relativeBinningGeneratedAt,
    relativeBinnedGeneratedAt,
    scalingSampleSize,
    binningSampleSize,
    binnedSampleSize,
  };
};
