import { render, screen, within } from '@testing-library/react';
import { expect, it } from 'vitest';
import { VariableStatistics } from './variableRow/VariableStatistics';
import { OutliersTab } from './tabs/OutliersTab';
import { OverviewCards } from './OverviewCards';
import type { EDAProfile } from '../../core/types/edaProfile';

it('does not invent zero variance for a singleton numeric profile', () => {
    // The real one-row analyzer emits null sample standard deviation and variance.
    render(<VariableStatistics profile={{ name: 'value', dtype: 'Numeric', missing_count: 0, missing_percentage: 0,
        numeric_stats: { mean: 4.2, std: null, variance: null, skewness: null, kurtosis: null } }} />);
    for (const name of ['Std Dev', 'Variance', 'Skewness', 'Kurtosis']) {
        expect(within(screen.getByText(name).parentElement!).getByText('—')).toBeVisible();
    }
    expect(screen.queryByText('0.0000')).not.toBeInTheDocument();
});

it('renders a real serialized overflowing outlier percentage without crashing', () => {
    // 99 values of 1e-308 plus 10 produce a null diff_pct after JSON serialization.
    const profile = { outliers: { method: 'IsolationForest', total_outliers: 1, outlier_percentage: 1,
        top_outliers: [{ index: 99, values: { tiny: 10 }, explanation: [
            { feature: 'tiny', value: 10, median: 1e-308, diff_pct: null },
        ] }],
    } } as unknown as EDAProfile;
    render(<OutliersTab profile={profile} />);
    expect(screen.getByText(/Diff: —/)).toBeVisible();
    expect(screen.getAllByText('10.00').length).toBeGreaterThan(0);
});

it('preserves measured zero variance and the legacy finite standard-deviation fallback', () => {
    // A missing field in an old report differs from an explicitly unavailable statistic.
    const column = { name: 'value', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 };
    const { rerender } = render(<VariableStatistics profile={{ ...column, numeric_stats: { variance: 0, std: 0 } }} />);
    expect(screen.getByText('Variance').parentElement).toHaveTextContent('0.0000');
    rerender(<VariableStatistics profile={{ ...column, numeric_stats: { std: 2 } }} />);
    expect(screen.getByText('Variance').parentElement).toHaveTextContent('4.0000');
});

it('does not interpret legacy unknown VIF as high collinearity', () => {
    // The existing finite 999 marker is high; serialized unknown values remain unknown.
    render(<OverviewCards profile={{ row_count: 2, column_count: 3, vif: { unknown: null, collinear: 999, low: 1 } } as unknown as EDAProfile} />);
    expect(screen.getByText('High VIF Features').parentElement).toHaveTextContent('High VIF Features1');
});
