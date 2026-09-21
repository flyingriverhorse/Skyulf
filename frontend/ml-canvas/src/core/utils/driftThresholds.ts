import type { DriftThresholds } from '../api/monitoring';

export const DEFAULT_DRIFT_THRESHOLDS: DriftThresholds = { psi: 0.2, ks: 0.1, wasserstein: 0.1, kl: 0.1 };

/** Distance/divergence thresholds are unbounded; the KS statistic is in [0, 1]. */
export function driftThresholdError(key: keyof DriftThresholds, value: number): string | null {
    if (!Number.isFinite(value) || value < 0) return 'Threshold must be finite and nonnegative.';
    if (key === 'ks' && value > 1) return 'KS threshold must be between 0 and 1.';
    return null;
}

/** Validate before FormData serialization, including callers outside the editor. */
export function validateDriftThresholds(thresholds: DriftThresholds): void {
    for (const key of Object.keys(thresholds) as (keyof DriftThresholds)[]) {
        const value = thresholds[key];
        if (value == null) continue;
        const error = driftThresholdError(key, value);
        if (error) throw new Error(error);
    }
}
