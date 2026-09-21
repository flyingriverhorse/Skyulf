import { fireEvent, render, screen } from '@testing-library/react';
import type { PropsWithChildren } from 'react';
import { expect, it, vi } from 'vitest';
import type { TimeSeriesAnalysis } from '../../../core/types/edaProfile';
import { TimeSeriesTab } from './TimeSeriesTab';

vi.mock('recharts', async importOriginal => ({
  ...await importOriginal<typeof import('recharts')>(),
  ResponsiveContainer: ({ children }: PropsWithChildren) => <div>{children}</div>,
  BarChart: ({ children }: PropsWithChildren) => <div>{children}</div>,
  Bar: ({ name, dataKey }: { name: string; dataKey: string }) => <span data-testid="bar" data-key={dataKey}>{name}</span>,
}));

it.each([
  [{ aggregation: 'mean', metric: 'sales' }, 'Mean of sales'],
  [{ aggregation: 'count', metric: null }, 'Row count'],
  [{}, 'Recorded value (measure unavailable)'],
] as const)('labels saved seasonality metadata %j without guessing from count', (metadata, label) => {
  /** Both bars, tooltips and exported charts must describe the stored statistic. */
  const timeseries: TimeSeriesAnalysis = {
    date_col: 'date', trend: [], seasonality: {
      day_of_week: [{ day: 'Mon', count: 20 }], month_of_year: [{ month: 'Jan', count: 20 }],
      ...metadata,
    },
  };
  const download = vi.fn();
  render(<TimeSeriesTab profile={{ timeseries }} downloadChart={download} />);
  expect(screen.getAllByTestId('bar')).toHaveLength(2);
  for (const bar of screen.getAllByTestId('bar')) {
    expect(bar).toHaveTextContent(label);
    expect(bar).toHaveAttribute('data-key', 'count');
  }
  fireEvent.click(screen.getAllByRole('button', { name: 'Download Chart' })[1]!);
  expect(download).toHaveBeenLastCalledWith('day-seasonality-chart', 'day-seasonality', 'Day of Week Seasonality', label);
  fireEvent.click(screen.getAllByRole('button', { name: 'Download Chart' })[2]!);
  expect(download).toHaveBeenLastCalledWith('month-seasonality-chart', 'month-seasonality', 'Monthly Seasonality', label);
});
