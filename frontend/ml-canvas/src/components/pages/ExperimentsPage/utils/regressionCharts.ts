/**
 * Pure helpers for the regression evaluation charts on the Experiments page.
 * No React, no DOM, no fetch — fully unit-testable.
 */

/** Q-Q plot: sorts residuals and maps them to theoretical normal quantiles. */
export function getQQData(y_true: number[], y_pred: number[]): { theoretical: number; sample: number }[] {
  const residuals = y_true.map((y, i) => y - (y_pred[i] ?? 0));
  const n = residuals.length;
  if (n < 2) return [];
  const mean = residuals.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(residuals.reduce((a, r) => a + (r - mean) ** 2, 0) / n);
  if (std === 0) return [];
  const sorted = [...residuals].sort((a, b) => a - b);
  // Standard normal inverse — Beasley-Springer-Moro approximation.
  const normInv = (p: number): number => {
    const a = [0, -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const b = [0, -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pLow = 0.02425, pHigh = 1 - pLow;
    if (p <= 0) return -8; if (p >= 1) return 8;
    if (p < pLow) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1); }
    if (p <= pHigh) { const q = p - 0.5, r = q * q; return (((((a[1]! * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * r + a[6]!) * q / (((((b[1]! * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + b[5]!) * r + 1); }
    const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  };
  return sorted.map((r, i) => ({
    theoretical: normInv((i + 0.5) / n) * std + mean,
    sample: r,
  }));
}

/** P50, P90, P95 absolute-error percentile chips. */
export function getErrorPercentiles(y_true: number[], y_pred: number[]): { p50: number; p90: number; p95: number } | null {
  const errs = y_true.map((y, i) => Math.abs(y - (y_pred[i] ?? 0))).sort((a, b) => a - b);
  if (errs.length === 0) return null;
  const pct = (q: number) => errs[Math.min(Math.floor(q * errs.length), errs.length - 1)]!;
  return { p50: pct(0.5), p90: pct(0.9), p95: pct(0.95) };
}

/** Relative error histogram: (pred − actual) / |actual|, binned. Skips near-zero actuals. */
export function getRelativeErrorHist(y_true: number[], y_pred: number[], nBins = 20): { label: string; count: number }[] | null {
  const relErrs = y_true.map((y, i) => Math.abs(y) > 1e-9 ? (y_pred[i]! - y) / Math.abs(y) : null).filter((v): v is number => v !== null);
  if (relErrs.length < 2) return null;
  const min = Math.max(Math.min(...relErrs), -2), max = Math.min(Math.max(...relErrs), 2);
  if (min === max) return null;
  const w = (max - min) / nBins;
  const bins = Array.from({ length: nBins }, (_, i) => ({ label: (min + i * w).toFixed(2), count: 0 }));
  for (const r of relErrs) { if (r >= min && r <= max) bins[Math.min(Math.floor((r - min) / w), nBins - 1)]!.count++; }
  return bins;
}

/** Scale-Location: sqrt(|residual|) vs predicted — surfaces heteroscedasticity. */
export function getScaleLocationData(y_true: number[], y_pred: number[]): { predicted: number; sqrtAbsRes: number }[] {
  return y_true.map((y, i) => ({ predicted: y_pred[i]!, sqrtAbsRes: Math.sqrt(Math.abs(y - y_pred[i]!)) }));
}

/** Sorted actual vs predicted: samples ordered by true value, useful for spotting tail divergence. */
export function getSortedActualPred(y_true: number[], y_pred: number[], maxPoints = 200): { idx: number; actual: number; predicted: number }[] {
  const pairs = y_true.map((y, i) => ({ actual: y, predicted: y_pred[i]! }));
  pairs.sort((a, b) => a.actual - b.actual);
  const step = Math.max(1, Math.floor(pairs.length / maxPoints));
  return pairs.filter((_, i) => i % step === 0 || i === pairs.length - 1).map((p, idx) => ({ idx, actual: p.actual, predicted: p.predicted }));
}

/** Residual lag plot: residual[i] vs residual[i-1] — surfaces serial correlation. */
export function getResidualLagData(y_true: number[], y_pred: number[]): { r0: number; r1: number }[] {
  const residuals = y_true.map((y, i) => y - y_pred[i]!);
  return residuals.slice(1).map((r, i) => ({ r0: residuals[i]!, r1: r }));
}

/** Residual histogram: bins of (actual − predicted), plus the mean residual. */
export function getResidualHistogram(
  y_true: number[],
  y_pred: number[],
  nBins = 20,
): { bins: { label: string; count: number }[]; mean: number } | null {
  const residuals = y_true.map((y, i) => y - (y_pred[i] ?? 0));
  if (residuals.length === 0) return null;
  const min = Math.min(...residuals), max = Math.max(...residuals);
  if (min === max) return null;
  const binWidth = (max - min) / nBins;
  const bins = Array.from({ length: nBins }, (_, i) => ({
    label: (min + i * binWidth).toFixed(2),
    count: 0,
  }));
  for (const r of residuals) {
    bins[Math.min(Math.floor((r - min) / binWidth), nBins - 1)]!.count++;
  }
  const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  return { bins, mean };
}
