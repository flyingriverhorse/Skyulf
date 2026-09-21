import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from './client';
import { thresholdTuningApi } from './thresholdTuning';
import { deploymentApi } from './deployment';
import { applyMulticlassThresholds } from '../../components/pages/ExperimentsPage/utils/classificationCharts';

const split = {
  y_true: [2, 0, 1], y_pred: [2, 0, 1],
  y_proba: { classes: [2, 0, 1], values: [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] },
};
const payload = { classes: [2, 0, 1], metric: 'f1', split_used: 'validation' };

describe('threshold denominator contract', () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([0, -0.1, NaN, Infinity, -Infinity])('rejects invalid multiclass denominator %s before drawing, saving or inference', async value => {
    /** Invalid scaled scores must produce errors rather than plausible predictions or JSON nulls. */
    const thresholds = { '0': 0.5, '1': value, '2': 0.5 };
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    expect(() => applyMulticlassThresholds(split, thresholds)).toThrow(/finite.*greater than 0/i);
    await expect(thresholdTuningApi.save('job', { ...payload, thresholds })).rejects.toThrow(/finite.*greater than 0/i);
    await expect(deploymentApi.predict([{ x: 1 }], thresholds)).rejects.toThrow(/finite.*greater than 0/i);
    expect(post).not.toHaveBeenCalled();
  });

  it('rejects incomplete or foreign class keys instead of silently substituting one', async () => {
    /** Every denominator must belong to the exact estimator class set. */
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    const invalidMaps: Record<string, number>[] = [
      { '0': 0.5, '1': 0.5 }, { '0': 0.5, '1': 0.5, '2': 0.5, extra: 1 },
    ];
    for (const thresholds of invalidMaps) {
      expect(() => applyMulticlassThresholds(split, thresholds)).toThrow(/every class/i);
      await expect(thresholdTuningApi.save('job', { ...payload, thresholds })).rejects.toThrow(/every class/i);
    }
    expect(post).not.toHaveBeenCalled();
  });

  it('rejects invalid preview output before it can enter chart state', async () => {
    /** A backend or legacy invalid threshold set must surface through the existing preview error UI. */
    vi.spyOn(apiClient, 'post').mockResolvedValue({ data: { ...payload, thresholds: { '0': 0.5, '1': 0, '2': 0.5 } } });
    await expect(thresholdTuningApi.preview('job', 'f1')).rejects.toThrow(/finite.*greater than 0/i);
  });

  it('accepts optimized denominators above one and preserves estimator order', async () => {
    /** Log-space optimization can legitimately produce thresholds above one. */
    const thresholds = { '0': 3, '1': 4, '2': 2 };
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    expect(applyMulticlassThresholds(split, thresholds)).toEqual({ classes: [2, 0, 1], matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] });
    await thresholdTuningApi.save('job', { ...payload, thresholds });
    expect(post).toHaveBeenCalledWith('/pipeline/jobs/job/thresholds/save', { ...payload, thresholds });
  });

  it.each([0, 1])('preserves binary boundary %s including exact zero/one probabilities', async threshold => {
    /** Binary endpoints follow positive probability >= threshold without division by zero. */
    const binary = { y_true: ['no', 'yes'], y_pred: ['no', 'yes'], y_proba: { classes: ['no', 'yes'], values: [[1, 0], [0, 1]] } };
    const thresholds = { no: 1 - threshold, yes: threshold };
    const result = applyMulticlassThresholds(binary, thresholds);
    expect(result.matrix).toEqual(threshold === 0 ? [[0, 1], [0, 1]] : [[1, 0], [0, 1]]);
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    await deploymentApi.predict([{ x: 1 }], thresholds);
    expect(post).toHaveBeenCalled();
  });
});
