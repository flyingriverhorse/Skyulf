import { describe, expect, it } from 'vitest';

import { getNodeMetricDetails, hasWrappedNodeMetrics } from './preprocessingMetrics';

describe('getNodeMetricDetails', () => {
  it('returns the single step details for wrapped preprocessing metrics', () => {
    expect(
      getNodeMetricDetails({
        fit_time: 0.3,
        summary: { fit_time: 0.3 },
        steps: {
          '0:step': {
            name: 'step',
            transformer: 'SimpleImputer',
            details: {
              fill_values: { a: 1.5 },
              total_missing: 2,
            },
          },
        },
      }),
    ).toEqual({
      fill_values: { a: 1.5 },
      total_missing: 2,
    });
  });

  it('returns null for ambiguous multi-step payloads without an explicit selector', () => {
    expect(
      getNodeMetricDetails({
        steps: {
          '0:impute': {
            details: { fill_values: { a: 1.5 } },
          },
          '1:scale': {
            details: { columns: ['a'], mean: [1] },
          },
        },
      }),
    ).toBeNull();
  });

  it('returns the selected step details for multi-step payloads', () => {
    expect(
      getNodeMetricDetails(
        {
          steps: {
            '0:impute': {
              details: { fill_values: { a: 1.5 } },
            },
            '1:scale': {
              details: { columns: ['a'], mean: [1] },
            },
          },
        },
        { stepKey: '1:scale' },
      ),
    ).toEqual({
      columns: ['a'],
      mean: [1],
    });
  });

  it('falls back to legacy flat metrics when steps are absent', () => {
    expect(
      getNodeMetricDetails({
        fill_values: { a: 1.5 },
        total_missing: 2,
      }),
    ).toEqual({
      fill_values: { a: 1.5 },
      total_missing: 2,
    });
  });

  it('returns null for non-object metrics payloads', () => {
    expect(getNodeMetricDetails(null)).toBeNull();
    expect(getNodeMetricDetails('bad payload')).toBeNull();
  });
});

describe('hasWrappedNodeMetrics', () => {
  it('detects wrapped metrics payloads', () => {
    expect(
      hasWrappedNodeMetrics({
        steps: {
          '0:step': {
            details: {},
          },
        },
      }),
    ).toBe(true);
  });

  it('returns false for legacy flat or invalid payloads', () => {
    expect(hasWrappedNodeMetrics({ dropped_columns: ['a'] })).toBe(false);
    expect(hasWrappedNodeMetrics(null)).toBe(false);
  });
});
