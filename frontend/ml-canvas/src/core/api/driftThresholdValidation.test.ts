import { afterEach, expect, it, vi } from 'vitest';
import { monitoringApi } from './monitoring';
import { apiClient } from './client';

afterEach(() => vi.restoreAllMocks());

it.each([{ psi: -1 }, { psi: Infinity }, { kl: NaN }, { wasserstein: -1 }, { ks: 1.01 }])('rejects invalid drift overrides before upload: %j', async thresholds => {
    // Bypassing the panel must not serialize invalid metric domains.
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    await expect(monitoringApi.calculateDrift('job', new File([''], 'data.csv'), undefined, thresholds)).rejects.toThrow(/threshold/i);
    expect(post).not.toHaveBeenCalled();
});
