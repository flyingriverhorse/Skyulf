import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
    monitoringApi,
    type ColumnDrift,
    type DriftMetric,
    type DriftReport,
    type DriftThresholds,
} from '../../../core/api/monitoring';
import { useDriftReport } from './useDriftReport';

vi.mock('../../../core/api/monitoring', async () => {
    const actual = await vi.importActual<typeof import('../../../core/api/monitoring')>(
        '../../../core/api/monitoring',
    );
    return {
        ...actual,
        monitoringApi: { ...actual.monitoringApi, calculateDrift: vi.fn() },
    };
});

const DEFAULTS: DriftThresholds = { psi: 0.2, ks: 0.1, wasserstein: 0.1, kl: 0.1 };

function column(name: string, metrics: DriftMetric[]): ColumnDrift {
    return {
        column: name,
        metrics,
        drift_detected: metrics.some(m => m.has_drift),
        suggestions: [],
    };
}

function report(
    columnDrifts: Record<string, ColumnDrift>,
    overrides: Partial<DriftReport> = {},
): DriftReport {
    return {
        reference_rows: 100,
        current_rows: 100,
        drifted_columns_count: 0,
        column_drifts: columnDrifts,
        missing_columns: [],
        new_columns: [],
        severity: 'none',
        ...overrides,
    };
}

/** Mounts the hook with a report already loaded from a mocked backend. */
async function loadReport(loaded: DriftReport) {
    vi.mocked(monitoringApi.calculateDrift).mockResolvedValue(loaded);
    const hook = renderHook(
        ({ thresholds }: { thresholds: DriftThresholds }) => useDriftReport(thresholds),
        { initialProps: { thresholds: DEFAULTS } },
    );
    await act(async () => {
        await hook.result.current.calculate({
            selectedJob: 'job-1',
            file: new File([''], 'current.csv'),
            job: undefined,
            thresholds: DEFAULTS,
        });
    });
    return hook;
}

describe('useDriftReport threshold re-evaluation', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('decides wasserstein on the threshold-scale value, not the raw distance', async () => {
        // A large-scale column: 50 units of earth-mover distance is only 0.017
        // reference standard deviations, so the backend decided "no drift" and
        // reported the normalized figure in `value`, keeping 50.0 in
        // `raw_value`. Comparing the raw one against the 0.1 slider inverts
        // that verdict (OC-44).
        const loaded = report({
            revenue: column('revenue', [
                {
                    metric: 'wasserstein_distance',
                    value: 0.017321,
                    has_drift: false,
                    threshold: 0.1,
                    raw_value: 50.0,
                },
            ]),
        });
        const { result, rerender } = await loadReport(loaded);

        const verdict = () => {
            const wd = result.current.evaluatedReport?.column_drifts.revenue?.metrics.find(
                m => m.metric === 'wasserstein_distance',
            );
            expect(wd, 'expected a wasserstein metric in the evaluated report').toBeDefined();
            return wd?.has_drift;
        };

        expect(verdict()).toBe(false);

        // Lowering the slider under the normalized distance must flip it, so the
        // comparison is genuinely on `value` rather than passing by accident.
        rerender({ thresholds: { ...DEFAULTS, wasserstein: 0.01 } });
        await waitFor(() => expect(verdict()).toBe(true));
    });

    it('keeps schema drift in the count when a threshold slider moves', async () => {
        // OC-45: the backend counts a vanished column as drift. The hook
        // rebuilds the count from the per-metric flags, which silently dropped
        // that contribution the moment a slider moved — leaving a report the
        // page called critical while showing zero drifted columns.
        const loaded = report(
            {
                stable: column('stable', [
                    { metric: 'psi', value: 0.01, has_drift: false, threshold: 0.2 },
                ]),
            },
            { drifted_columns_count: 1, missing_columns: ['dropped'], severity: 'critical' },
        );
        const { result, rerender } = await loadReport(loaded);

        const count = () => result.current.evaluatedReport?.drifted_columns_count;
        expect(count()).toBe(1);

        rerender({ thresholds: { ...DEFAULTS, psi: 0.5 } });
        await waitFor(() => expect(count()).toBe(1));
    });

    it('counts distribution drift alongside schema drift', async () => {
        const loaded = report(
            {
                shifted: column('shifted', [
                    { metric: 'psi', value: 0.6, has_drift: true, threshold: 0.2 },
                ]),
                stable: column('stable', [
                    { metric: 'psi', value: 0.01, has_drift: false, threshold: 0.2 },
                ]),
            },
            {
                drifted_columns_count: 3,
                missing_columns: ['dropped'],
                new_columns: ['added'],
                severity: 'critical',
            },
        );
        const { result } = await loadReport(loaded);

        expect(result.current.evaluatedReport?.drifted_columns_count).toBe(3);
    });
});
