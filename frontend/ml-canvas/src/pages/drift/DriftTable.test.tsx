import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { ColumnDrift, DriftReport } from '../../core/api/monitoring';
import { DriftTable } from './DriftTable';

/** Stable row flags must remain independent of unusual metric display values. */
function fixture(): DriftReport {
    const columns: ColumnDrift[] = [
        { column: 'zebra', drift_detected: true, suggestions: ['Check source'], metrics: [
            { metric: 'psi_categorical', value: 0.3, threshold: 0.2, has_drift: true },
            { metric: 'ks_statistic', value: 0.2, threshold: 0.1, has_drift: true },
            { metric: 'ks_test_p_value', value: 0.0001, threshold: 0.1, has_drift: true },
        ] },
        { column: 'alpha', drift_detected: false, suggestions: [], metrics: [
            { metric: 'wasserstein_distance', value: 0, raw_value: 0, threshold: 0.1, has_drift: false },
            { metric: 'psi', value: NaN, threshold: 0.2, has_drift: false },
            { metric: 'kl_divergence', value: Infinity, threshold: 0.1, has_drift: true },
        ] },
    ];
    return { reference_rows: 100, current_rows: 50, drifted_columns_count: 2, severity: 'critical',
        missing_columns: ['gone'], new_columns: [], column_drifts: Object.fromEntries(columns.map(c => [c.column, c])),
        feature_importances: { zebra: 0.8, alpha: 0.2 } };
}

describe('DriftTable public evidence', () => {
    it('preserves ordering, special values, metric tooltips, details and report reset', () => {
        /** Sorting and expansion must preserve metric evidence and collapse for a new report. */
        const onSort = vi.fn();
        const props = { report: fixture(), showOnlyDrifted: false, sortConfig: { key: 'column', dir: 'asc' as const }, onSort, columnSparklines: {} };
        const { rerender } = render(<DriftTable {...props} />);
        const rows = screen.getAllByRole('row');
        expect(within(rows[1]!).getByText('alpha')).toBeInTheDocument();
        expect(screen.getByText('NaN')).toBeInTheDocument();
        expect(screen.getByText('Infinity')).toBeInTheDocument();
        expect(screen.getByTitle('Raw earth-mover distance: 0.0000 in column units')).toHaveTextContent('0.0000');
        fireEvent.click(screen.getByText('Column'));
        expect(onSort).toHaveBeenCalledWith('column');
        fireEvent.click(within(rows[2]!).getByRole('button', { name: 'Details' }));
        expect(screen.getByText('< 0.001')).toBeInTheDocument();
        expect(screen.getByText('Check source')).toBeInTheDocument();
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(screen.queryByText('Check source')).not.toBeInTheDocument();
        fireEvent.click(within(rows[2]!).getByRole('button', { name: 'Details' }));
        rerender(<DriftTable {...props} report={fixture()} showOnlyDrifted />);
        expect(screen.queryByText('alpha')).not.toBeInTheDocument();
        expect(screen.queryByText('Check source')).not.toBeInTheDocument();
        expect(screen.getByText('0.3000')).toHaveClass('font-bold');
    });

    it('distinguishes schema-only drift from a stable filtered table', () => {
        /** A missing shared row must not hide structural drift. */
        const report = { ...fixture(), column_drifts: {} };
        const props = { report, showOnlyDrifted: true, sortConfig: null, onSort: vi.fn(), columnSparklines: {} };
        const { rerender } = render(<DriftTable {...props} />);
        expect(screen.getByText(/see the schema drift above/)).toBeInTheDocument();
        rerender(<DriftTable {...props} report={{ ...report, missing_columns: [] }} />);
        expect(screen.getByText(/all features are stable/)).toBeInTheDocument();
    });
});
