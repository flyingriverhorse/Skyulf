import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { DriftReport } from '../../core/api/monitoring';
import { SummaryCards } from './SummaryCards';

/** A saved report may count schema changes alongside the four measured features. */
function makeReport(drifted: boolean, schemaChanges = true): DriftReport {
    return {
        reference_rows: 120,
        current_rows: 100,
        drifted_columns_count: (drifted ? 4 : 0) + (schemaChanges ? 2 : 0),
        column_drifts: Object.fromEntries(['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'].map(name => [name, {
            column: name,
            drift_detected: drifted,
            metrics: [{ metric: 'psi', value: drifted ? 0.8 : 0, has_drift: drifted, threshold: 0.2 }],
            suggestions: [],
        }])),
        missing_columns: schemaChanges ? [''] : [],
        new_columns: schemaChanges ? ['Species'] : [],
        severity: drifted || schemaChanges ? 'critical' : 'none',
    };
}

describe('SummaryCards feature drift percentage', () => {
    it('does not include schema changes in the percentage of measured features', () => {
        // Historic 6/4 reports must display the four feature verdicts as 100%.
        render(<SummaryCards report={makeReport(true)} />);

        expect(screen.queryByText('150% of features')).not.toBeInTheDocument();
        expect(screen.getByText('100% of features')).toBeInTheDocument();
    });

    it('keeps stable feature distributions at zero when only the schema changes', () => {
        // A structural alert must not invent value drift in stable measurements.
        render(<SummaryCards report={makeReport(false)} />);

        expect(screen.getByText('0% of features')).toBeInTheDocument();
    });

    it('handles a report without comparable features', () => {
        // Disjoint schemas must not cause a divide-by-zero percentage.
        render(<SummaryCards report={{ ...makeReport(false), column_drifts: {} }} />);

        expect(screen.getByText('0% of features')).toBeInTheDocument();
    });
});
