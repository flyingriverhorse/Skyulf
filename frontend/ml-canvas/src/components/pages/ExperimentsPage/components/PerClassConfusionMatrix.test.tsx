// Unit tests for the Task 7 tuned-threshold redraw wiring in
// `PerClassConfusionMatrix`: when `useTunedThresholds && tunedThresholds` is
// truthy, the per-split matrix must be recomputed via
// `applyMulticlassThresholds` instead of the existing `applyThreshold`
// (single-class OvR) path — and the existing path must be completely
// unchanged when the new props are omitted or `useTunedThresholds` is false.

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PerClassConfusionMatrix } from './PerClassConfusionMatrix';
import type { EvaluationData } from '../types';

// 3 classes so the component doesn't early-return (it renders nothing for
// binary — `y_proba.classes.length <= 2`).
const evaluationData: Extract<EvaluationData, { problem_type: 'classification' | 'regression' }> = {
  problem_type: 'classification' as const,
  splits: {
    train: {
      // Deliberately "wrong" original predictions (always 'a') so the
      // untuned path is trivially distinguishable from the tuned-threshold
      // recomputed argmax below.
      y_true: ['a', 'b', 'c', 'a'],
      y_pred: ['a', 'a', 'a', 'a'],
      y_proba: {
        classes: ['a', 'b', 'c'],
        values: [
          [0.5, 0.3, 0.2], // argmax a
          [0.2, 0.5, 0.3], // argmax b
          [0.3, 0.3, 0.4], // argmax c
          [0.4, 0.35, 0.25], // argmax a
        ],
      },
    },
  },
};

const noop = async () => {};

describe('PerClassConfusionMatrix — tuned threshold redraw', () => {
  it('uses the existing applyThreshold path when the new props are omitted', () => {
    render(
      <PerClassConfusionMatrix
        evaluationData={evaluationData}
        selectedRocClass={null}
        threshold={0.5}
        showTrainMetrics
        showTestMetrics={false}
        showValMetrics={false}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    // Original y_pred is all 'a', so class 'a' has 2 true positives (both
    // 'a' samples correctly predicted) and class 'b'/'c' both have 0 TP.
    const aPanel = screen.getByTitle('a vs Rest').parentElement!;
    expect(aPanel.querySelector('[title="TP=2"]')).not.toBeNull();
    const bPanel = screen.getByTitle('b vs Rest').parentElement!;
    expect(bPanel.querySelector('[title="TP=0"]')).not.toBeNull();
  });

  it('ignores tunedThresholds when useTunedThresholds is false (both conditions required)', () => {
    render(
      <PerClassConfusionMatrix
        evaluationData={evaluationData}
        selectedRocClass={null}
        threshold={0.5}
        showTrainMetrics
        showTestMetrics={false}
        showValMetrics={false}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
        tunedThresholds={{ a: 1, b: 1, c: 1 }}
        useTunedThresholds={false}
      />,
    );
    const aPanel = screen.getByTitle('a vs Rest').parentElement!;
    expect(aPanel.querySelector('[title="TP=2"]')).not.toBeNull();
  });

  it('redraws via applyMulticlassThresholds when useTunedThresholds && tunedThresholds are both truthy', () => {
    render(
      <PerClassConfusionMatrix
        evaluationData={evaluationData}
        selectedRocClass={null}
        threshold={0.5}
        showTrainMetrics
        showTestMetrics={false}
        showValMetrics={false}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
        tunedThresholds={{ a: 1, b: 1, c: 1 }}
        useTunedThresholds
      />,
    );
    // Equal thresholds (all 1) reduce applyMulticlassThresholds to plain
    // argmax over y_proba, which perfectly matches y_true (['a','b','c','a']
    // vs argmax predictions ['a','b','c','a']) — every class gets 1 TP with
    // no false negatives, unlike the "always predict a" default path above.
    const aPanel = screen.getByTitle('a vs Rest').parentElement!;
    expect(aPanel.querySelector('[title="TP=2"]')).not.toBeNull();
    const bPanel = screen.getByTitle('b vs Rest').parentElement!;
    expect(bPanel.querySelector('[title="TP=1"]')).not.toBeNull();
    const cPanel = screen.getByTitle('c vs Rest').parentElement!;
    expect(cPanel.querySelector('[title="TP=1"]')).not.toBeNull();
  });
});
