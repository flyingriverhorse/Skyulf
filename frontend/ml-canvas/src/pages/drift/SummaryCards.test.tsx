import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ColumnDrift, DriftReport } from '../../core/api/monitoring';
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

/** Mirror the per-column metric names preserved by the calculate endpoint. */
function psiColumn(column: string, metric: string, value: number): ColumnDrift {
    return {
        column, drift_detected: value > 0.2, suggestions: [],
        metrics: [{ metric, value, has_drift: value > 0.2, threshold: 0.2 }],
    };
}

/** Keep schema accounting independent of the PSI values under test. */
function psiReport(columns: ColumnDrift[]): DriftReport {
    return {
        reference_rows: 100, current_rows: 100, missing_columns: [], new_columns: [],
        severity: 'critical', drifted_columns_count: columns.filter(column => column.drift_detected).length,
        column_drifts: Object.fromEntries(columns.map(column => [column.column, column])),
    };
}

/** Scope assertions to the headline card, where the misleading summary appeared. */
function summaryCard(label: string) {
    return within(screen.getByText(label, { exact: true }).parentElement!);
}

describe('SummaryCards PSI evidence', () => {
    it('includes a dominant categorical shift in the average and most drifted column', () => {
        /** A stable numeric feature must not hide the categorical evidence returned by the API. */
        render(<SummaryCards report={psiReport([
            psiColumn('numeric', 'psi', 0.01), psiColumn('category', 'psi_categorical', 5),
        ])} />);

        expect(summaryCard('Avg PSI').getByText('2.5050')).toBeInTheDocument();
        expect(summaryCard('Avg PSI').getByText('Significant drift')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('category')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('PSI: 5.0000')).toBeInTheDocument();
        expect(screen.getByText('50% of features')).toBeInTheDocument();
    });

    it.each([
        [0, '0.0000', 'Stable'],
        [0.15, '0.1500', 'Minor drift'],
        [0.3, '0.3000', 'Significant drift'],
    ])('summarizes a categorical-only PSI of %s', (value, formatted, interpretation) => {
        /** Categorical-only reports need the same numeric summary and bands as numeric reports. */
        render(<SummaryCards report={psiReport([psiColumn('category', 'psi_categorical', value)])} />);

        expect(summaryCard('Avg PSI').getByText(formatted)).toBeInTheDocument();
        expect(summaryCard('Avg PSI').getByText(interpretation)).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText(`PSI: ${formatted}`)).toBeInTheDocument();
    });

    it('excludes unavailable metrics while retaining measured zero in the average', () => {
        /** Missing PSI must not dilute measured drift, and zero must remain valid evidence. */
        render(<SummaryCards report={psiReport([
            psiColumn('no-psi', 'ks_statistic', 0.7),
            psiColumn('not-a-number', 'psi', NaN),
            psiColumn('infinite', 'psi_categorical', Infinity),
            psiColumn('negative-infinite', 'psi', -Infinity),
            psiColumn('zero', 'psi_categorical', 0),
            psiColumn('numeric', 'psi', 0.4),
        ])} />);

        expect(summaryCard('Avg PSI').getByText('0.2000')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('numeric')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('PSI: 0.4000')).toBeInTheDocument();
    });

    it('ranks a measured zero above columns without PSI', () => {
        /** A missing score must not win a tie against a real stable measurement. */
        render(<SummaryCards report={psiReport([
            psiColumn('unavailable', 'ks_statistic', 0.7), psiColumn('zero', 'psi_categorical', 0),
        ])} />);

        expect(summaryCard('Avg PSI').getByText('0.0000')).toBeInTheDocument();
        expect(summaryCard('Avg PSI').getByText('Stable')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('zero')).toBeInTheDocument();
    });

    it.each(['empty', 'missing', 'non-finite', 'json-null'])('shows unavailable PSI for %s evidence', kind => {
        /** An absent measurement must not be displayed as zero or labelled stable. */
        const columns = kind === 'empty' ? [] : [
            psiColumn('unavailable', kind === 'missing' ? 'ks_statistic' : 'psi_categorical', NaN),
        ];
        const report: DriftReport = kind === 'json-null'
            ? JSON.parse(JSON.stringify(psiReport(columns))) : psiReport(columns);
        render(<SummaryCards report={report} />);

        expect(summaryCard('Avg PSI').getByText('—')).toBeInTheDocument();
        expect(summaryCard('Avg PSI').getByText('No PSI available')).toBeInTheDocument();
        expect(summaryCard('Avg PSI').queryByText('Stable')).not.toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('—')).toBeInTheDocument();
        expect(summaryCard('Most Drifted').getByText('PSI: —')).toBeInTheDocument();
    });
});
