import { describe, it, expect } from 'vitest';
import {
  DEFAULT_EXECUTION_MODE,
  EXECUTION_MODE_AWARE_TYPES,
  AUTO_PARALLEL_TYPES,
  supportsExecutionModeToggle,
  isAutoParallelType,
  getExecutionMode,
  isParallelExecution,
  getNodeExecutionMode,
} from './executionMode';

describe('executionMode helpers', () => {
  it('exposes the canonical default of "merge"', () => {
    expect(DEFAULT_EXECUTION_MODE).toBe('merge');
  });

  it('lists the modeling-node toggle types', () => {
    expect(EXECUTION_MODE_AWARE_TYPES.has('classification')).toBe(true);
    expect(EXECUTION_MODE_AWARE_TYPES.has('regression')).toBe(true);
    // A non-modeling node must NOT report toggle support, otherwise the
    // Properties Panel would render an unusable mode selector.
    expect(EXECUTION_MODE_AWARE_TYPES.has('imputation_node')).toBe(false);
  });

  it('lists data_preview as auto-parallel (mirrors backend AUTO_PARALLEL_STEP_TYPES)', () => {
    expect(AUTO_PARALLEL_TYPES.has('data_preview')).toBe(true);
  });

  it('supportsExecutionModeToggle / isAutoParallelType narrow correctly', () => {
    expect(supportsExecutionModeToggle('classification')).toBe(true);
    expect(supportsExecutionModeToggle('data_preview')).toBe(false);
    expect(isAutoParallelType('data_preview')).toBe(true);
    expect(isAutoParallelType('classification')).toBe(false);
  });

  it('getExecutionMode falls back to "merge" for missing/invalid input', () => {
    expect(getExecutionMode(undefined)).toBe('merge');
    expect(getExecutionMode(null)).toBe('merge');
    expect(getExecutionMode({})).toBe('merge');
    expect(getExecutionMode({ execution_mode: 'parallel' })).toBe('parallel');
    // An unknown literal must not be trusted; default back to merge so the
    // engine never receives a value it can't dispatch.
    expect(getExecutionMode({ execution_mode: 'bogus' as unknown as 'merge' })).toBe('merge');
  });

  it('isParallelExecution: training nodes need the explicit toggle', () => {
    expect(
      isParallelExecution({ definitionType: 'classification', execution_mode: 'parallel' }, 2),
    ).toBe(true);
    expect(
      isParallelExecution({ definitionType: 'classification', execution_mode: 'merge' }, 2),
    ).toBe(false);
  });

  it('supportsExecutionModeToggle recognizes canonical task-scoped node types', () => {
    expect(supportsExecutionModeToggle('classification')).toBe(true);
    expect(supportsExecutionModeToggle('regression')).toBe(true);
    expect(supportsExecutionModeToggle('text_classification')).toBe(true);
    expect(supportsExecutionModeToggle('training')).toBe(true);
  });

  it('isParallelExecution: auto-parallel terminals need 2+ upstream sources', () => {
    expect(isParallelExecution({ definitionType: 'data_preview' }, 2)).toBe(true);
    // One upstream source = nothing to fan out into; treat as merge.
    expect(isParallelExecution({ definitionType: 'data_preview' }, 1)).toBe(false);
  });

  it('getNodeExecutionMode reads from node.data', () => {
    expect(
      getNodeExecutionMode({
        data: { execution_mode: 'parallel' },
      }),
    ).toBe('parallel');
  });
});
