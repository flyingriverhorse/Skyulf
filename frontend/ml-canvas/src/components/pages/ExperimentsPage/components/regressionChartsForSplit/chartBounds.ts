import type { getResidualHistogram, getQQData } from '../../utils/regressionCharts';

/** Position the mean marker at the closest histogram bin. */
export function getMeanBinIndex(hist: ReturnType<typeof getResidualHistogram>) {
  return hist
    ? hist.bins.reduce((best, b, i) => (
      Math.abs(Number.parseFloat(b.label) - hist.mean) < Math.abs(Number.parseFloat(hist.bins[best]!.label) - hist.mean) ? i : best
    ), 0)
    : 0;
}

/** Use one common axis domain for the sample and theoretical quantiles. */
export function getQQBounds(qqData: ReturnType<typeof getQQData>) {
  const qqMin = qqData.length ? Math.min(...qqData.map(d => Math.min(d.theoretical, d.sample))) : 0;
  const qqMax = qqData.length ? Math.max(...qqData.map(d => Math.max(d.theoretical, d.sample))) : 1;
  return { qqMin, qqMax };
}
