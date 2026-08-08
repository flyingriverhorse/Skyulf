import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DistributionChart } from './DistributionChart';
import type { ColumnProfile } from '../../core/types/edaProfile';

const numericProfile: ColumnProfile = {
  name: 'age',
  dtype: 'Numeric',
  missing_count: 0,
  missing_percentage: 0,
  histogram: [
    { start: 0, end: 10, count: 4 },
    { start: 10, end: 20, count: 9 },
  ],
};

describe('DistributionChart', () => {
  it('provides a data table alternative with the same bins and counts as the chart', () => {
    render(<DistributionChart profile={numericProfile} />);

    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    const region = screen.getByRole('region', { name: /distribution data table for age/i });

    expect(within(region).getByText('0.00 - 10.00')).toBeInTheDocument();
    expect(within(region).getByText('9')).toBeInTheDocument();
  });

  it('shows an explanatory message rather than a blank chart when the type is unsupported', () => {
    render(<DistributionChart profile={{ ...numericProfile, dtype: 'Boolean', histogram: null }} />);
    expect(screen.getByText(/no distribution data available/i)).toBeInTheDocument();
  });
});
