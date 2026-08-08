import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { CorrelationHeatmap } from './CorrelationHeatmap';

const smallData = {
  columns: ['sepal_length', 'sepal_width'],
  values: [
    [1, -0.12],
    [-0.12, 1],
  ],
};

const manyColumns = Array.from({ length: 25 }, (_, i) => `feature_${i}`);
const manyValues = manyColumns.map((_, i) => manyColumns.map((_, j) => (i === j ? 1 : 0.1)));

afterEach(() => {
  document.documentElement.classList.remove('dark');
});

describe('CorrelationHeatmap', () => {
  it('renders a persistent -1/0/+1 color scale legend, not just colored cells', () => {
    render(<CorrelationHeatmap data={smallData} />);
    const legend = screen.getByRole('img', { name: /correlation color scale/i });
    expect(legend).toBeInTheDocument();
    expect(screen.getByText('−1')).toBeInTheDocument();
    expect(screen.getByText('+1')).toBeInTheDocument();
  });

  it('renders full column labels in the DOM, not only via a hover-only title attribute', () => {
    render(<CorrelationHeatmap data={smallData} />);
    // The full name must appear as visible text content somewhere, not only as a `title`.
    expect(screen.getAllByText('sepal_length').length).toBeGreaterThan(0);
    expect(screen.getAllByText('sepal_width').length).toBeGreaterThan(0);
  });

  it('names the omitted columns and count when truncating a large matrix', () => {
    render(<CorrelationHeatmap data={{ columns: manyColumns, values: manyValues }} />);
    expect(screen.getByText(/showing the first 20 of 25 columns/i)).toBeInTheDocument();
    expect(screen.getByText(/5 omitted/i)).toBeInTheDocument();
  });

  it('offers a full data-table alternative that includes truncated columns', () => {
    render(<CorrelationHeatmap data={{ columns: manyColumns, values: manyValues }} />);
    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));

    const region = screen.getByRole('region', { name: /full correlation matrix/i });
    // feature_24 is beyond the MAX_COLS=20 visual truncation but must still appear in the table.
    expect(within(region).getAllByText('feature_24').length).toBeGreaterThan(0);
  });

  it('keeps correlation values legible as text regardless of theme', () => {
    document.documentElement.classList.add('dark');
    render(<CorrelationHeatmap data={smallData} />);
    expect(screen.getAllByText('1.00').length).toBeGreaterThan(0);
    expect(screen.getAllByText('-0.12').length).toBeGreaterThan(0);
  });

  it('shows a no-data message when there is no correlation data', () => {
    render(<CorrelationHeatmap data={undefined as unknown as { columns: string[]; values: number[][] }} />);
    expect(screen.getByText(/no correlation data available/i)).toBeInTheDocument();
  });
});
