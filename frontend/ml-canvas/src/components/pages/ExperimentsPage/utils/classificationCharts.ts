/**
 * Pure helpers for the classification evaluation charts on the Experiments page.
 * No React, no DOM, no fetch — fully unit-testable.
 *
 * Convention: `targetClass` is treated as the "positive" class for One-vs-Rest views.
 * String coercion is intentional because <select> values are always strings.
 */

import type { EvaluationSplit, YProba } from '../types';

/** Locate the column index of `targetClass` in y_proba, preferring `labels` over `classes`. */
function findClassIndex(y_proba: YProba, targetClass: string | number): number {
  const targetStr = String(targetClass);
  const labelList = y_proba.labels?.length === y_proba.classes.length ? y_proba.labels : undefined;
  return (labelList ?? y_proba.classes).findIndex(c => String(c) === targetStr);
}

/** Confusion matrix for an arbitrary set of classes. classOrder forces row/column order. */
export function calculateConfusionMatrix(
  y_true: (string | number)[],
  y_pred: (string | number)[],
  classOrder?: (string | number)[],
): { classes: (string | number)[]; matrix: number[][] } {
  const classes = (classOrder && classOrder.length > 0)
    ? [...classOrder]
    : Array.from(new Set([...y_true, ...y_pred])).sort((a, b) => String(a).localeCompare(String(b)));
  const matrix = classes.map(trueClass =>
    classes.map(predClass =>
      y_true.reduce((count: number, t, i) => {
        const p = y_pred[i];
        return (String(t) === String(trueClass) && String(p) === String(predClass)) ? count + 1 : count;
      }, 0),
    ),
  );
  return { classes, matrix };
}

/**
 * Reassigns predicted classes for one split using an OvR (one-vs-rest) decision
 * threshold on `selectedClass`, then returns the resulting confusion matrix.
 *
 * Shared by ClassificationChartsForSplit and PerClassConfusionMatrix, which
 * both need the same "apply threshold to reassign predictions" behaviour —
 * previously duplicated verbatim in both components.
 *
 * A sample is predicted as `selectedClass` when P(selectedClass) >= threshold;
 * otherwise it falls back to the argmax of all other classes' probabilities.
 * If `selectedClass` is not provided, or `y_proba` is missing, no threshold
 * reassignment is applied and the split's original y_true/y_pred are used.
 */
export function applyThreshold(
  splitData: EvaluationSplit,
  selectedClass: string | null,
  threshold: number,
): { classes: (string | number)[]; matrix: number[][] } {
  const proba = splitData.y_proba;
  let yTrue: (string | number)[] = splitData.y_true;
  let yPred: (string | number)[] = splitData.y_pred;

  // Map string labels back to their class values, when the split reports
  // labels separately from classes (e.g. label-encoded targets).
  if (proba?.labels && proba.labels.length === proba.classes.length) {
    const labelToClass = new Map<string, string | number>();
    proba.labels.forEach((label, idx) => {
      const cls = proba.classes[idx];
      if (cls !== undefined) labelToClass.set(String(label), cls);
    });
    yTrue = yTrue.map(y => labelToClass.get(String(y)) ?? y);
    yPred = yPred.map(y => labelToClass.get(String(y)) ?? y);
  }

  // Apply OvR threshold for the selected class (works for binary and multiclass)
  if (proba && selectedClass) {
    const labelList = proba.labels?.length === proba.classes.length ? proba.labels : undefined;
    const posIdx = (labelList ?? proba.classes).findIndex(c => String(c) === selectedClass);
    if (posIdx !== -1) {
      const posVal = proba.classes[posIdx];
      const origPred = [...yPred];
      if (posVal !== undefined) {
        yPred = proba.values.map((v, i) => {
          if ((v[posIdx] ?? 0) >= threshold) return posVal;
          // Argmax of all other classes
          let bestIdx = -1, bestProb = -Infinity;
          v.forEach((p, idx) => {
            if (idx !== posIdx && p > bestProb) { bestProb = p; bestIdx = idx; }
          });
          return bestIdx >= 0 ? (proba.classes[bestIdx] ?? origPred[i]!) : (origPred[i]!);
        });
      }
    }
  }

  return calculateConfusionMatrix(yTrue, yPred, proba?.classes);
}

/** Metric that a best-threshold scan can optimize for. `f1`/`precision`/`recall`
 * are the binary positive-class ("bare") metrics — matching the backend's own
 * naming in `_add_binary_unweighted_metrics` — and fall back to their
 * support-weighted multiclass equivalent when the split has >2 classes, since
 * the unweighted binary form doesn't exist there. `accuracy` and `f1_weighted`
 * are well-defined for both. */
export type ThresholdMetric = 'accuracy' | 'f1' | 'f1_weighted' | 'precision' | 'recall';

/**
 * The dropdown's selectable metric list for a given class count. Binary jobs
 * show the plain positive-class metrics (Accuracy/F1/Precision/Recall) since
 * "weighted" vs "bare" are genuinely different numbers there. Multiclass
 * jobs show `f1_weighted` instead of `f1` — for multiclass there's no
 * unweighted form, so showing both would just repeat the same number twice
 * under two different labels. Precision/Recall are the same underlying
 * `ThresholdMetric` value in both cases; only their *label* changes (see
 * `metricLabel` below) since the multiclass computation is already the
 * weighted average.
 */
export function thresholdMetricOptions(isBinary: boolean): ThresholdMetric[] {
  return isBinary
    ? ['accuracy', 'f1', 'precision', 'recall']
    : ['accuracy', 'f1_weighted', 'precision', 'recall'];
}

/**
 * Human-readable label for a metric, aware of whether Precision/Recall/F1
 * mean the plain positive-class value (binary) or the support-weighted
 * multiclass average — so the dropdown/badges never show a bare label next
 * to a number that was actually computed as a weighted average.
 */
export function metricLabel(metric: ThresholdMetric, isBinary: boolean): string {
  if (isBinary) {
    return { accuracy: 'Accuracy', f1: 'F1', f1_weighted: 'F1 Weighted', precision: 'Precision', recall: 'Recall' }[metric];
  }
  return { accuracy: 'Accuracy', f1: 'F1 Weighted', f1_weighted: 'F1 Weighted', precision: 'Precision (Weighted)', recall: 'Recall (Weighted)' }[metric];
}

/**
 * Coerces a metric selection to one that's actually offered for the given
 * class count — e.g. a binary job's saved `f1` selection becomes
 * `f1_weighted` when switching to view a multiclass job (same underlying
 * value either way is fine computationally, this just keeps the <select>'s
 * value in sync with its visible <option> list).
 */
export function normalizeThresholdMetric(metric: ThresholdMetric, isBinary: boolean): ThresholdMetric {
  if (isBinary) return metric === 'f1_weighted' ? 'f1' : metric;
  return metric === 'f1' ? 'f1_weighted' : metric;
}

interface BinaryCounts { tp: number; fp: number; fn: number; tn: number; }

/** Derives the requested metric from 2x2 confusion counts for the positive class. */
function binaryMetricValue(counts: BinaryCounts, metric: ThresholdMetric): number {
  const { tp, fp, fn, tn } = counts;
  const total = tp + fp + fn + tn;
  if (total === 0) return 0;
  if (metric === 'accuracy') return (tp + tn) / total;
  const precisionPos = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recallPos = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1Pos = (precisionPos + recallPos) > 0 ? (2 * precisionPos * recallPos) / (precisionPos + recallPos) : 0;
  if (metric === 'precision') return precisionPos;
  if (metric === 'recall') return recallPos;
  if (metric === 'f1') return f1Pos;
  // f1_weighted: support-weighted average of the pos/neg per-class F1 (mirrors sklearn's average='weighted')
  const precisionNeg = (tn + fn) > 0 ? tn / (tn + fn) : 0;
  const recallNeg = (tn + fp) > 0 ? tn / (tn + fp) : 0;
  const f1Neg = (precisionNeg + recallNeg) > 0 ? (2 * precisionNeg * recallNeg) / (precisionNeg + recallNeg) : 0;
  const supportPos = tp + fn, supportNeg = tn + fp;
  return (supportPos + supportNeg) > 0 ? (f1Pos * supportPos + f1Neg * supportNeg) / (supportPos + supportNeg) : 0;
}

/** Derives the requested metric from a full multiclass confusion matrix (support-weighted average). */
function multiclassMetricValue(
  classes: (string | number)[],
  matrix: number[][],
  metric: ThresholdMetric,
): number {
  const k = classes.length;
  const total = matrix.reduce((s, row) => s + row.reduce((rs, v) => rs + v, 0), 0);
  if (total === 0) return 0;
  if (metric === 'accuracy') {
    let correct = 0;
    for (let i = 0; i < k; i++) correct += matrix[i]?.[i] ?? 0;
    return correct / total;
  }
  let weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;
  for (let i = 0; i < k; i++) {
    const tp = matrix[i]?.[i] ?? 0;
    let fp = 0, fn = 0, support = 0;
    for (let j = 0; j < k; j++) {
      if (j !== i) fp += matrix[j]?.[i] ?? 0;
      support += matrix[i]?.[j] ?? 0;
    }
    fn = support - tp;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    weightedPrecision += precision * support;
    weightedRecall += recall * support;
    weightedF1 += f1 * support;
  }
  if (metric === 'precision') return weightedPrecision / total;
  if (metric === 'recall') return weightedRecall / total;
  // `f1` and `f1_weighted` are numerically identical here — multiclass has no
  // unweighted "bare" F1, so both fall back to the same weighted-average value.
  return weightedF1 / total;
}

/**
 * Best classification threshold for `targetClass`: scans every unique
 * prediction score and returns the one maximising the requested `metric`.
 *
 * Binary splits use a fast O(n) per-candidate scan (tp/fp/fn/tn counters).
 * Multiclass splits reuse `applyThreshold`'s OvR-reassignment + confusion
 * matrix (the same logic already used to render the confusion matrix chart),
 * capped to ~300 sampled candidate thresholds to bound the O(n² · k) cost on
 * large validation sets.
 */
export function findBestThreshold(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
  metric: ThresholdMetric,
): { threshold: number; value: number } | null {
  const classIdx = findClassIndex(y_proba, targetClass);
  if (classIdx === -1) return null;
  const targetStr = String(targetClass);
  const scores = y_proba.values.map(v => v[classIdx] ?? 0);
  const isBinary = y_proba.classes.length === 2;

  if (isBinary) {
    const actual = y_true.map(y => String(y) === targetStr ? 1 : 0);
    if (!actual.some(a => a === 1)) return null;
    const candidates = [...new Set(scores)].sort((a, b) => a - b);
    let bestValue = -1, bestT = 0.5;
    for (const t of candidates) {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (let i = 0; i < scores.length; i++) {
        const pred = (scores[i]! >= t) ? 1 : 0;
        if (pred === 1 && actual[i] === 1) tp++;
        else if (pred === 1 && actual[i] === 0) fp++;
        else if (pred === 0 && actual[i] === 1) fn++;
        else tn++;
      }
      const value = binaryMetricValue({ tp, fp, fn, tn }, metric);
      if (value > bestValue) { bestValue = value; bestT = t; }
    }
    return { threshold: Math.round(bestT * 100) / 100, value: bestValue };
  }

  if (!y_true.some(y => String(y) === targetStr)) return null;
  let candidates = [...new Set(scores)].sort((a, b) => a - b);
  const MAX_CANDIDATES = 300;
  if (candidates.length > MAX_CANDIDATES) {
    const step = (candidates.length - 1) / (MAX_CANDIDATES - 1);
    candidates = Array.from({ length: MAX_CANDIDATES }, (_, i) => candidates[Math.round(i * step)]!);
  }
  const pseudoSplit: EvaluationSplit = { y_true, y_pred: y_true, y_proba };
  let bestValue = -1, bestT = 0.5;
  for (const t of candidates) {
    const { classes, matrix } = applyThreshold(pseudoSplit, targetStr, t);
    const value = multiclassMetricValue(classes, matrix, metric);
    if (value > bestValue) { bestValue = value; bestT = t; }
  }
  return { threshold: Math.round(bestT * 100) / 100, value: bestValue };
}

/** ROC points (FPR, TPR) for one class vs all others. Returns null if either side is empty in the split. */
export function calculateROC(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
): { fpr: number; tpr: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const scores = y_proba.values.map(v => v[classIndex] ?? 0);
  const data = scores.map((score, i) => ({
    score,
    actual: String(y_true[i]) === targetStr ? 1 : 0,
  }));
  data.sort((a, b) => b.score - a.score);
  const totalPos = data.filter(d => d.actual === 1).length;
  const totalNeg = data.length - totalPos;
  if (totalPos === 0 || totalNeg === 0) return null;
  const rocPoints: { fpr: number; tpr: number }[] = [{ fpr: 0, tpr: 0 }];
  let tp = 0, fp = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]!.actual === 1) tp++;
    else fp++;
    rocPoints.push({ fpr: fp / totalNeg, tpr: tp / totalPos });
  }
  return rocPoints;
}

/** Precision-Recall curve for one class vs all others. Points sorted by score (descending). */
export function calculatePR(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
): { recall: number; precision: number; score: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const scores = y_proba.values.map(v => v[classIndex] ?? 0);
  const items = scores.map((score, i) => ({
    score,
    actual: String(y_true[i]) === targetStr ? 1 : 0,
  }));
  items.sort((a, b) => b.score - a.score);
  const totalPos = items.filter(d => d.actual === 1).length;
  if (totalPos === 0) return null;
  const points: { recall: number; precision: number; score: number }[] = [{ recall: 0, precision: 1, score: 1 }];
  let tp = 0, fp = 0;
  for (const item of items) {
    if (item.actual === 1) tp++; else fp++;
    points.push({ recall: tp / totalPos, precision: tp / (tp + fp), score: item.score });
  }
  return points;
}

/** Overlapping score-distribution histogram of `P(targetClass)`, split into pos/neg buckets. */
export function getScoreDistribution(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
  nBins = 20,
): { range: string; pos: number; neg: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const bins = Array.from({ length: nBins }, (_, i) => ({
    range: `${(i / nBins).toFixed(2)}`,
    pos: 0,
    neg: 0,
  }));
  for (let i = 0; i < y_proba.values.length; i++) {
    const score = y_proba.values[i]![classIndex] ?? 0;
    const binIdx = Math.min(Math.floor(score * nBins), nBins - 1);
    if (String(y_true[i]) === targetStr) bins[binIdx]!.pos++;
    else bins[binIdx]!.neg++;
  }
  return bins;
}

/** Calibration curve: bins model confidence vs actual fraction of positives. Empty bins are dropped. */
export function getCalibrationData(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
  nBins = 10,
): { midpoint: number; fracPos: number; count: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const bins: { pos: number; count: number }[] = Array.from({ length: nBins }, () => ({ pos: 0, count: 0 }));
  for (let i = 0; i < y_proba.values.length; i++) {
    const score = y_proba.values[i]![classIndex] ?? 0;
    const binIdx = Math.min(Math.floor(score * nBins), nBins - 1);
    bins[binIdx]!.pos += String(y_true[i]) === targetStr ? 1 : 0;
    bins[binIdx]!.count++;
  }
  return bins
    .map((b, i) => ({
      midpoint: parseFloat(((i + 0.5) / nBins).toFixed(2)),
      fracPos: b.count > 0 ? parseFloat((b.pos / b.count).toFixed(4)) : 0,
      count: b.count,
    }))
    .filter(b => b.count > 0);
}

/** Cumulative gains: fraction of samples targeted (sorted by score desc) vs fraction of positives captured. */
export function getCumulativeGainsData(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
): { pct: number; gain: number; lift: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const items = y_proba.values.map((v, i) => ({
    score: v[classIndex] ?? 0,
    actual: String(y_true[i]) === targetStr ? 1 : 0,
  }));
  items.sort((a, b) => b.score - a.score);
  const totalPos = items.filter(d => d.actual === 1).length;
  if (totalPos === 0) return null;
  const n = items.length;
  const pts: { pct: number; gain: number; lift: number }[] = [{ pct: 0, gain: 0, lift: 1 }];
  let cumPos = 0;
  for (let i = 0; i < n; i++) {
    cumPos += items[i]!.actual;
    const pct = (i + 1) / n;
    const gain = cumPos / totalPos;
    pts.push({
      pct: parseFloat(pct.toFixed(3)),
      gain: parseFloat(gain.toFixed(3)),
      lift: parseFloat((gain / pct).toFixed(3)),
    });
  }
  // Down-sample to keep the chart responsive on large splits.
  const step = Math.max(1, Math.floor(pts.length / 60));
  return pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
}

/** MCC (Matthews Correlation Coefficient) sweep across `nSteps + 1` thresholds in [0, 1]. */
export function getMCCByThreshold(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
  nSteps = 50,
): { threshold: number; mcc: number }[] | null {
  const classIndex = findClassIndex(y_proba, targetClass);
  if (classIndex === -1) return null;
  const targetStr = String(targetClass);
  const scores = y_proba.values.map(v => v[classIndex] ?? 0);
  const actual = y_true.map(t => (String(t) === targetStr ? 1 : 0));
  return Array.from({ length: nSteps + 1 }, (_, s) => {
    const t = s / nSteps;
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < scores.length; i++) {
      const pred = (scores[i]! >= t) ? 1 : 0;
      if (pred === 1 && actual[i] === 1) tp++;
      else if (pred === 1 && actual[i] === 0) fp++;
      else if (pred === 0 && actual[i] === 0) tn++;
      else fn++;
    }
    const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    return {
      threshold: parseFloat(t.toFixed(2)),
      mcc: parseFloat((denom > 0 ? (tp * tn - fp * fn) / denom : 0).toFixed(4)),
    };
  });
}
