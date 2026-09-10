import type { ThresholdMetric } from '../classificationCharts';

interface BinaryCounts { tp: number; fp: number; fn: number; tn: number; }

/** Derives the requested metric from 2x2 confusion counts for the positive class. */
function binaryMetricValue(counts: BinaryCounts, metric: ThresholdMetric): number {
  const { tp, fp, fn, tn } = counts;
  const total = tp + fp + fn + tn;
  if(total === 0) return 0;
  if(metric === 'accuracy') return (tp + tn) / total;
  const { precisionPos, recallPos, f1Pos } = positiveRates(tp, fp, fn);
  if(metric === 'precision') return precisionPos;
  if(metric === 'recall') return recallPos;
  if(metric === 'f1') return f1Pos;
  // f1_weighted: support-weighted average of the pos/neg per-class F1 (mirrors sklearn's average='weighted')
  const precisionNeg = (tn + fn) > 0 ? tn / (tn + fn) : 0;
  const recallNeg = (tn + fp) > 0 ? tn / (tn + fp) : 0;
  const f1Neg = (precisionNeg + recallNeg) > 0 ? (2 * precisionNeg * recallNeg) / (precisionNeg + recallNeg) : 0;
  const supportPos = tp + fn, supportNeg = tn + fp;
  return (supportPos + supportNeg) > 0 ? (f1Pos * supportPos + f1Neg * supportNeg) / (supportPos + supportNeg) : 0;
}

/** Derives the requested metric from a full multiclass confusion matrix (support-weighted average). */
export function multiclassMetricValue(
  classes: (string | number)[],
  matrix: number[][],
  metric: ThresholdMetric,
): number {
  const k = classes.length;
  const total = matrix.reduce((s, row) => s + row.reduce((rs, v) => rs + v, 0), 0);
  if(total === 0) return 0;
  if(metric === 'accuracy') {
    let correct = 0;
    for(let i = 0;i < k;i++) correct += matrix[i]?.[i] ?? 0;
    return correct / total;
  }
  let weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;
  for(let i = 0;i < k;i++) {
    const { precision, recall, f1, support } = classRates(matrix, i, k);
    weightedPrecision += precision * support;
    weightedRecall += recall * support;
    weightedF1 += f1 * support;
  }
  if(metric === 'precision') return weightedPrecision / total;
  if(metric === 'recall') return weightedRecall / total;
  // `f1` and `f1_weighted` are numerically identical here — multiclass has no
  // unweighted "bare" F1, so both fall back to the same weighted-average value.
  return weightedF1 / total;
}

/** Positive-class precision, recall and harmonic mean. */
function positiveRates(tp: number, fp: number, fn: number) {
  const precisionPos = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recallPos = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1Pos = (precisionPos + recallPos) > 0 ? (2 * precisionPos * recallPos) / (precisionPos + recallPos) : 0;
  return { precisionPos, recallPos, f1Pos };
}

/** Support and OvR rates for one row of a multiclass matrix. */
function classRates(matrix: number[][], i: number, k: number) {
  const tp = matrix[i]?.[i] ?? 0;
  let fp = 0, fn = 0, support = 0;
  for(let j = 0;j < k;j++) {
    if(j !== i) fp += matrix[j]?.[i] ?? 0;
    support += matrix[i]?.[j] ?? 0;
  }
  fn = support - tp;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { precision, recall, f1, support };
}

/** Scan every binary candidate, preserving the first threshold on ties. */
export function findBinaryThreshold(y_true: (string | number)[], scores: number[], targetStr: string, metric: ThresholdMetric) {
  const actual = y_true.map(y => String(y) === targetStr ? 1 : 0);
  if(!actual.some(a => a === 1)) return null;
  const candidates = [...new Set(scores)].sort((a, b) => a - b);
  let bestValue = -1, bestT = 0.5;
  for(const t of candidates) {
    const { tp, fp, fn, tn } = countBinaryPredictions(scores, actual, t);
    const value = binaryMetricValue({ tp, fp, fn, tn }, metric);
    if(value > bestValue) { bestValue = value; bestT = t; }
  }
  return { threshold: Math.round(bestT * 100) / 100, value: bestValue };
}

/** Count thresholded binary predictions without allocating a matrix. */
function countBinaryPredictions(scores: number[], actual: number[], t: number): BinaryCounts {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for(let i = 0;i < scores.length;i++) {
    const pred = (scores[i]! >= t) ? 1 : 0;
    if(pred === 1 && actual[i] === 1) tp++;
    else if(pred === 1 && actual[i] === 0) fp++;
    else if(pred === 0 && actual[i] === 1) fn++;
    else tn++;
  }
  return { tp, fp, fn, tn };
}
