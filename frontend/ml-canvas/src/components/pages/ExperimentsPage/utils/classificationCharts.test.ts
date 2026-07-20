// Unit tests for the classification-evaluation best-threshold scan
// (`findBestThreshold`), added alongside the Model Evaluation "Metric:"
// dropdown feature (accuracy/f1/f1_weighted/precision/recall).

import { describe, it, expect } from 'vitest';
import { findBestThreshold } from './classificationCharts';
import type { YProba } from '../types';

describe('findBestThreshold — binary', () => {
  // 4 samples, P(pos) scores [0.9, 0.4, 0.6, 0.1], true labels [pos, pos, neg, neg].
  // Hand-derived optimum per metric (see design notes): f1/accuracy peak at
  // t=0.4, precision peaks at t=0.9, recall peaks at t=0.1 (first tie wins
  // since candidates are scanned ascending and `>` — not `>=` — replaces
  // the running best).
  const yTrue = ['pos', 'pos', 'neg', 'neg'];
  const proba: YProba = {
    classes: ['pos', 'neg'],
    values: [
      [0.9, 0.1],
      [0.4, 0.6],
      [0.6, 0.4],
      [0.1, 0.9],
    ],
  };

  it('optimizes f1', () => {
    const result = findBestThreshold(yTrue, proba, 'pos', 'f1');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.4, 5);
    expect(result!.value).toBeCloseTo(0.8, 5);
  });

  it('optimizes accuracy', () => {
    const result = findBestThreshold(yTrue, proba, 'pos', 'accuracy');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.4, 5);
    expect(result!.value).toBeCloseTo(0.75, 5);
  });

  it('optimizes precision (differs from f1/accuracy optimum)', () => {
    const result = findBestThreshold(yTrue, proba, 'pos', 'precision');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.9, 5);
    expect(result!.value).toBeCloseTo(1.0, 5);
  });

  it('optimizes recall (differs from precision optimum)', () => {
    const result = findBestThreshold(yTrue, proba, 'pos', 'recall');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.1, 5);
    expect(result!.value).toBeCloseTo(1.0, 5);
  });

  it('optimizes f1_weighted, distinct from bare f1 for binary jobs', () => {
    const result = findBestThreshold(yTrue, proba, 'pos', 'f1_weighted');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.4, 5);
    expect(result!.value).toBeCloseTo(0.7333, 3);
  });

  it('returns null when the target class never occurs', () => {
    const result = findBestThreshold(['neg', 'neg'], {
      classes: ['pos', 'neg'],
      values: [[0.5, 0.5], [0.3, 0.7]],
    }, 'pos', 'f1');
    expect(result).toBeNull();
  });
});

describe('findBestThreshold — multiclass', () => {
  // 3 classes; the OvR threshold on "A" is scanned across every unique
  // P(A) score, reusing the same confusion-matrix reassignment logic as
  // the rendered confusion matrix (`applyThreshold`).
  const yTrue = ['A', 'A', 'B', 'C'];
  const proba: YProba = {
    classes: ['A', 'B', 'C'],
    values: [
      [0.9, 0.05, 0.05],
      [0.4, 0.5, 0.1],
      [0.3, 0.6, 0.1],
      [0.2, 0.1, 0.7],
    ],
  };

  it('finds the perfectly-separating threshold for accuracy', () => {
    const result = findBestThreshold(yTrue, proba, 'A', 'accuracy');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeCloseTo(0.4, 5);
    expect(result!.value).toBeCloseTo(1.0, 5);
  });

  it('f1 and f1_weighted are numerically identical for multiclass (no unweighted multiclass F1)', () => {
    const f1 = findBestThreshold(yTrue, proba, 'A', 'f1');
    const f1Weighted = findBestThreshold(yTrue, proba, 'A', 'f1_weighted');
    expect(f1).not.toBeNull();
    expect(f1Weighted).not.toBeNull();
    expect(f1!.threshold).toBeCloseTo(f1Weighted!.threshold, 5);
    expect(f1!.value).toBeCloseTo(f1Weighted!.value, 5);
  });

  it('precision and recall fall back to the support-weighted multiclass average', () => {
    const precision = findBestThreshold(yTrue, proba, 'A', 'precision');
    const recall = findBestThreshold(yTrue, proba, 'A', 'recall');
    expect(precision).not.toBeNull();
    expect(recall).not.toBeNull();
    expect(precision!.value).toBeGreaterThanOrEqual(0);
    expect(precision!.value).toBeLessThanOrEqual(1);
    expect(recall!.value).toBeGreaterThanOrEqual(0);
    expect(recall!.value).toBeLessThanOrEqual(1);
  });

  it('returns null when the target class never occurs', () => {
    const result = findBestThreshold(['B', 'C'], proba, 'A', 'accuracy');
    expect(result).toBeNull();
  });
});

describe('findBestThreshold — large multiclass split (sampling cap)', () => {
  it('completes and returns a valid result when there are far more than 300 unique candidate scores', () => {
    const n = 2000;
    const classes = ['A', 'B', 'C'];
    const yTrue: (string | number)[] = [];
    const values: number[][] = [];
    for (let i = 0; i < n; i++) {
      const pA = (i % 997) / 997; // 997 is prime relative to n, so scores are all distinct
      const rest = (1 - pA) / 2;
      values.push([pA, rest, rest]);
      yTrue.push(pA > 0.5 ? 'A' : (i % 2 === 0 ? 'B' : 'C'));
    }
    const proba: YProba = { classes, values };
    const result = findBestThreshold(yTrue, proba, 'A', 'f1_weighted');
    expect(result).not.toBeNull();
    expect(result!.threshold).toBeGreaterThanOrEqual(0);
    expect(result!.threshold).toBeLessThanOrEqual(1);
    expect(result!.value).toBeGreaterThanOrEqual(0);
    expect(result!.value).toBeLessThanOrEqual(1);
  });
});
