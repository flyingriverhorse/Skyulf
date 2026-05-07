/**
 * Pure helpers for the classification evaluation charts on the Experiments page.
 * No React, no DOM, no fetch — fully unit-testable.
 *
 * Convention: `targetClass` is treated as the "positive" class for One-vs-Rest views.
 * String coercion is intentional because <select> values are always strings.
 */

import type { YProba } from '../types';

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

/** Best classification threshold: scans every unique prediction score and returns the one maximising F1 for targetClass. */
export function findBestF1Threshold(
  y_true: (string | number)[],
  y_proba: YProba,
  targetClass: string | number,
): { threshold: number; f1: number } | null {
  const classIdx = findClassIndex(y_proba, targetClass);
  if (classIdx === -1) return null;
  const targetStr = String(targetClass);
  const scores = y_proba.values.map(v => v[classIdx] ?? 0);
  const actual = y_true.map(y => String(y) === targetStr ? 1 : 0);
  if (!actual.some(a => a === 1)) return null;
  const candidates = [...new Set(scores)].sort((a, b) => a - b);
  let bestF1 = -1, bestT = 0.5;
  for (const t of candidates) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < scores.length; i++) {
      const pred = (scores[i]! >= t) ? 1 : 0;
      if (pred === 1 && actual[i] === 1) tp++;
      else if (pred === 1 && actual[i] === 0) fp++;
      else if (pred === 0 && actual[i] === 1) fn++;
    }
    const denom = 2 * tp + fp + fn;
    const f1 = denom > 0 ? (2 * tp) / denom : 0;
    if (f1 > bestF1) { bestF1 = f1; bestT = t; }
  }
  return { threshold: Math.round(bestT * 100) / 100, f1: bestF1 };
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
