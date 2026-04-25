import { describe, it, expect } from 'vitest';
import { asBaseNodeData, getMergeStrategy } from './nodeData';

describe('nodeData helpers', () => {
  it('asBaseNodeData narrows objects without crashing on null/undefined', () => {
    expect(asBaseNodeData(undefined)).toEqual({});
    expect(asBaseNodeData(null)).toEqual({});
    expect(asBaseNodeData({ foo: 1 })).toEqual({ foo: 1 });
  });

  it('getMergeStrategy returns engine default "last_wins" when absent', () => {
    expect(getMergeStrategy(undefined)).toBe('last_wins');
    expect(getMergeStrategy({})).toBe('last_wins');
  });

  it('getMergeStrategy returns the explicit strategy when set', () => {
    expect(getMergeStrategy({ merge_strategy: 'first_wins' })).toBe('first_wins');
  });
});
