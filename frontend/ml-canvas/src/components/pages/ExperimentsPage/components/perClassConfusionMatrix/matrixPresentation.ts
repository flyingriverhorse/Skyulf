import type { EvaluationSplit } from '../../types';

/** Preserve train/test/validation ordering independently of payload ordering. */
export function getVisibleSplits(splits: Record<string, EvaluationSplit>, showTrainMetrics: boolean, showTestMetrics: boolean, showValMetrics: boolean) {
  const allSplitEntries = Object.entries(splits) as [string, EvaluationSplit][];
  const trainEntry = showTrainMetrics ? allSplitEntries.find(([n]) => n === 'train') : undefined;
  const testEntry = showTestMetrics ? allSplitEntries.find(([n]) => n === 'test') : undefined;
  const valEntry = showValMetrics ? allSplitEntries.find(([n]) => n === 'validation') : undefined;
  return { trainEntry, testEntry, valEntry };
}

/** Counts and rates used by a per-class one-versus-rest panel. */
export function getClassRates(matrix: number[][], clsIdx: number) {
  const tp = matrix[clsIdx]?.[clsIdx] ?? 0;
  const fp = matrix.reduce((s, row, ri) => ri !== clsIdx ? s + (row[clsIdx] ?? 0) : s, 0);
  const fn = (matrix[clsIdx] ?? []).reduce((s, v, ci) => ci !== clsIdx ? s + v : s, 0);
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const tn = total - tp - fp - fn;
  const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1c = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  return { tp, fp, fn, tn, prec, rec, f1c };
}
