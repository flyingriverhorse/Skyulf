import { describe, expect, it } from 'vitest';

import {
  partitionSelection,
  resolveEvaluationTarget,
  selectRunsForView,
  type SelectableRun,
} from './runSelection';

const run = (jobId: string, task: SelectableRun['task'], visible: boolean): SelectableRun => ({
  jobId,
  task,
  visible,
});

describe('partitionSelection', () => {
  it('separates selected runs the active filter hides from the ones it shows', () => {
    const runs = [
      run('a', 'classification', false),
      run('b', 'segmentation', true),
      run('c', 'classification', false),
    ];
    expect(partitionSelection(runs)).toEqual({ visible: ['b'], hidden: ['a', 'c'] });
  });

  it('reports no hidden runs when the filter shows every selection', () => {
    const runs = [run('a', 'regression', true), run('b', 'regression', true)];
    expect(partitionSelection(runs)).toEqual({ visible: ['a', 'b'], hidden: [] });
  });

  it('handles an empty selection', () => {
    expect(partitionSelection([])).toEqual({ visible: [], hidden: [] });
  });

  it('preserves selection order within each group', () => {
    const runs = [
      run('z', 'classification', false),
      run('y', 'classification', true),
      run('x', 'classification', false),
      run('w', 'classification', true),
    ];
    expect(partitionSelection(runs)).toEqual({ visible: ['y', 'w'], hidden: ['z', 'x'] });
  });
});

describe('resolveEvaluationTarget', () => {
  it('picks a clustering run for the segmentation tab instead of an incompatible one', () => {
    const runs = [
      run('cls', 'classification', false),
      run('km', 'segmentation', true),
    ];
    expect(resolveEvaluationTarget('segmentation', runs, 'cls')).toBe('km');
  });

  it('never targets a clustering run from the evaluation tab', () => {
    const runs = [run('km', 'segmentation', true), run('cls', 'classification', true)];
    expect(resolveEvaluationTarget('evaluation', runs, null)).toBe('cls');
  });

  it('keeps the current target when it is still selected and compatible', () => {
    const runs = [run('a', 'classification', true), run('b', 'classification', true)];
    expect(resolveEvaluationTarget('evaluation', runs, 'b')).toBe('b');
  });

  it('drops a target that is no longer selected', () => {
    const runs = [run('a', 'classification', true)];
    expect(resolveEvaluationTarget('evaluation', runs, 'gone')).toBe('a');
  });

  it('prefers a visible compatible run over a hidden one', () => {
    const runs = [
      run('hidden', 'classification', false),
      run('shown', 'classification', true),
    ];
    expect(resolveEvaluationTarget('evaluation', runs, null)).toBe('shown');
  });

  it('still targets a hidden run when no compatible run is visible', () => {
    const runs = [run('hidden', 'classification', false)];
    expect(resolveEvaluationTarget('evaluation', runs, null)).toBe('hidden');
  });

  it('returns null when no selected run suits the tab', () => {
    const runs = [run('km', 'segmentation', true)];
    expect(resolveEvaluationTarget('evaluation', runs, null)).toBeNull();
  });

  it('returns null for an empty selection', () => {
    expect(resolveEvaluationTarget('segmentation', [], 'stale')).toBeNull();
  });

  it('treats regression, text classification, and ensemble runs as evaluable', () => {
    expect(resolveEvaluationTarget('evaluation', [run('r', 'regression', true)], null)).toBe('r');
    expect(
      resolveEvaluationTarget('evaluation', [run('t', 'text_classification', true)], null),
    ).toBe('t');
    expect(resolveEvaluationTarget('evaluation', [run('e', 'ensemble', true)], null)).toBe('e');
  });
});

describe('selectRunsForView', () => {
  const mixed = [
    run('cls', 'classification', true),
    run('km', 'segmentation', true),
    run('reg', 'regression', false),
  ];

  it('offers only clustering runs to the segmentation tab', () => {
    expect(selectRunsForView('segmentation', mixed)).toEqual(['km']);
  });

  it('excludes clustering runs from the evaluation tab', () => {
    expect(selectRunsForView('evaluation', mixed)).toEqual(['cls', 'reg']);
  });

  it('keeps filter-hidden runs available so the picker matches the comparison', () => {
    expect(selectRunsForView('evaluation', [run('hidden', 'regression', false)])).toEqual([
      'hidden',
    ]);
  });

  it('returns an empty list when nothing suits the view', () => {
    expect(selectRunsForView('segmentation', [run('cls', 'classification', true)])).toEqual([]);
  });
});
