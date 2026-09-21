import { describe, expect, it } from 'vitest';
import { getRelativeErrorHist } from './regressionCharts';

describe('relative error population accounting', () => {
  it('counts both tails separately while retaining exact boundary values', () => {
    // Large model failures must not vanish from the displayed population.
    const result = getRelativeErrorHist([1, 1, 1, 1, 1], [-2, -1, 1, 3, 4])!;
    expect(result.bins[0]).toMatchObject({ label: '< -200%', count: 1 });
    expect(result.bins.at(-1)).toMatchObject({ label: '> 200%', count: 1 });
    expect(result.bins[1]).toMatchObject({ range: '[-200%, -180%)', count: 1 });
    expect(result.bins.at(-2)).toMatchObject({ range: '[180%, 200%]', count: 1 });
    expect(result.bins.reduce((total, bin) => total + bin.count, 0)).toBe(5);
    expect(result).toMatchObject({ included: 5, excluded: 0 });
  });

  it.each([{ errors: [-4, -3] }, { errors: [4, 5] }, { errors: [0, 0] }])('keeps an all-tail or constant population visible: $errors', ({ errors }) => {
    // Degenerate ranges must still produce a useful, nonempty histogram.
    const result = getRelativeErrorHist([1, 1], errors.map(error => error + 1))!;
    expect(result.bins.reduce((total, bin) => total + bin.count, 0)).toBe(2);
    expect(result.bins.every(bin => Number.isFinite(bin.count))).toBe(true);
  });

  it('accounts for near-zero actuals and missing or nonfinite pairs explicitly', () => {
    // Invalid ratios must not poison bins or be silently counted as valid errors.
    const result = getRelativeErrorHist([0, 1e-10, 1, 2, Infinity, 3], [1, 2, 2, NaN, 1])!;
    expect(result).toMatchObject({ included: 1, excluded: 5 });
    expect(result.bins.reduce((total, bin) => total + bin.count, 0)).toBe(1);
  });

  it('reports a fully excluded population and reserves null for no samples', () => {
    // A hidden chart must not conceal why no ratios could be computed.
    expect(getRelativeErrorHist([0, 1e-12], [1, 1])).toMatchObject({ included: 0, excluded: 2 });
    expect(getRelativeErrorHist([], [])).toBeNull();
  });
});
