import { describe, expect, it } from 'vitest';

import { buildFeatureBarChartData, reportedFlagKey, type FeatureBarJobEntry } from './featureBarChart';

const job = (overrides: Partial<FeatureBarJobEntry>): FeatureBarJobEntry => ({
  jobId: 'job-1',
  pipeline_id: 'preview_abcd1234efgh',
  parent_pipeline_id: null,
  modelType: 'random_forest',
  values: null,
  ...overrides,
});

describe('buildFeatureBarChartData', () => {
  it('normalizes each run so its largest-magnitude feature is 1.0', () => {
    const { chartData, barKeys } = buildFeatureBarChartData([
      job({ jobId: 'a', values: { feature_a: 4, feature_b: 2 } }),
    ]);
    const barKey = barKeys[0]!;
    const rowA = chartData.find((r) => r.feature === 'feature_a');
    const rowB = chartData.find((r) => r.feature === 'feature_b');
    expect(rowA?.[barKey]).toBe(1);
    expect(rowB?.[barKey]).toBe(0.5);
  });

  it('flags a feature a run does not report as unreported, distinct from a genuine zero', () => {
    const { chartData, barKeys } = buildFeatureBarChartData([
      job({ jobId: 'a', values: { feature_a: 1, feature_b: 0 } }),
      job({ jobId: 'b', modelType: 'gbm', values: { feature_a: 1 } }),
    ]);
    const barKeyA = barKeys[0]!;
    const barKeyB = barKeys[1]!;
    const rowB = chartData.find((r) => r.feature === 'feature_b');

    // Run "a" genuinely scored feature_b at 0 — reported.
    expect(rowB?.[reportedFlagKey(barKeyA)]).toBe(true);
    expect(rowB?.[barKeyA]).toBe(0);

    // Run "b" never scored feature_b at all — not reported, rendered as 0
    // only as a chart placeholder, but the flag says otherwise.
    expect(rowB?.[reportedFlagKey(barKeyB)]).toBe(false);
    expect(rowB?.[barKeyB]).toBe(0);
  });

  it('excludes jobs with null values (artifact not supported/computed) from the chart data but reports the count', () => {
    const { jobsWithDataCount, barKeys } = buildFeatureBarChartData([
      job({ jobId: 'a', values: { feature_a: 1 } }),
      job({ jobId: 'b', values: null }),
    ]);
    expect(jobsWithDataCount).toBe(1);
    expect(barKeys).toHaveLength(1);
  });

  it('ranks and truncates to the top 15 features by average normalized importance', () => {
    const values: Record<string, number> = {};
    for (let i = 0; i < 20; i++) {
      values[`feature_${i}`] = 20 - i;
    }
    const { topFeatures, allFeatures } = buildFeatureBarChartData([job({ values })]);
    expect(topFeatures).toHaveLength(15);
    expect(allFeatures.size).toBe(20);
    expect(topFeatures[0]).toBe('feature_0');
  });

  it('exposes the raw (non-normalized) values per bar key for the data table', () => {
    const { barKeys, rawByBarKey } = buildFeatureBarChartData([
      job({ jobId: 'a', values: { feature_a: 42 } }),
    ]);
    expect(rawByBarKey[barKeys[0]!]?.feature_a).toBe(42);
  });
});
