import { describe, expect, it } from 'vitest';

import { getNodeMetricDetails } from './preprocessingMetrics';

describe('getNodeMetricDetails', () => {
  it('returns the first step details for nested preprocessing metrics', () => {
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
