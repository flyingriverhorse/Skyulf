import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { OutliersTab } from './OutliersTab';
import type { EDAProfile } from '../../../core/types/edaProfile';

/** Keep the stored outlier counts constant so only their population context changes. */
function profile(population: Record<string, number | null> = {}): EDAProfile {
  return {
    row_count: 200000, column_count: 1, columns: {},
    outliers: {
      method: 'IsolationForest', total_outliers: 2500, outlier_percentage: 5,
      top_outliers: [{ index: 100001, value: 12, score: -0.1, values: { measurement: 12 } }],
      ...population,
    },
  };
}

describe('outlier analysis population', () => {
  it('labels counts and percentages as sample results with both row counts', () => {
    /** Sample results must not imply that every dataset row was scored. */
    render(<OutliersTab profile={profile({ analyzed_rows: 50000, total_rows: 200000 })} />);
    expect(screen.getByText(/Analyzed 50,000 sampled rows out of 200,000 rows/i)).toBeInTheDocument();
    expect(screen.getByText('Outliers in sample')).toBeInTheDocument();
    expect(screen.getByText('Percentage of sampled rows')).toBeInTheDocument();
    expect(screen.getByText('2500')).toBeInTheDocument();
    expect(screen.getByText('5.00%')).toBeInTheDocument();
    expect(screen.queryByText('Total Outliers')).not.toBeInTheDocument();
  });

  it('distinguishes complete analysis from sampled results using stored metadata', () => {
    /** Filtered analysis metadata takes precedence over a surrounding report count. */
    render(<OutliersTab profile={profile({ analyzed_rows: 50000, total_rows: 50000 })} />);
    expect(screen.getByText(/Analyzed all 50,000 rows/i)).toBeInTheDocument();
    expect(screen.getByText('Outliers detected')).toBeInTheDocument();
    expect(screen.getByText('Percentage of analyzed rows')).toBeInTheDocument();
    expect(screen.queryByText(/200,000/)).not.toBeInTheDocument();
  });

  it.each([{}, { analyzed_rows: null, total_rows: null }, { analyzed_rows: 50000 }, { total_rows: 200000 }])('does not infer a denominator for legacy reports %j', population => {
    /** Historic percentages cannot be safely converted into sampled or population row counts. */
    render(<OutliersTab profile={profile(population)} />);
    expect(screen.getByText(/Analyzed row count is unavailable for this saved report/i)).toBeInTheDocument();
    expect(screen.getByText('Reported outliers')).toBeInTheDocument();
    expect(screen.getByText('Reported percentage')).toBeInTheDocument();
    expect(screen.queryByText(/200,000/)).not.toBeInTheDocument();
  });
});
