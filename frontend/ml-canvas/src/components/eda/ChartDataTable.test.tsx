import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ChartDataTable } from './ChartDataTable';

const columns = [
  { key: 'name', label: 'Name' },
  { key: 'count', label: 'Count' },
];

const rows = [
  { name: 'a', count: 3 },
  { name: 'b', count: 5 },
];

describe('ChartDataTable', () => {
  it('is collapsed by default and reveals the table on toggle', () => {
    render(<ChartDataTable columns={columns} rows={rows} filename="test" caption="Test data" />);

    expect(screen.queryByRole('region', { name: 'Test data' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));

    const region = screen.getByRole('region', { name: 'Test data' });
    expect(within(region).getByText('a')).toBeInTheDocument();
    expect(within(region).getByText('5')).toBeInTheDocument();
  });

  it('renders every column header so no data is hover-only', () => {
    render(<ChartDataTable columns={columns} rows={rows} filename="test" caption="Test data" defaultOpen />);
    expect(screen.getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Count' })).toBeInTheDocument();
  });

  it('shows an explicit empty message instead of a blank table when there are no rows', () => {
    render(<ChartDataTable columns={columns} rows={[]} filename="test" caption="Test data" defaultOpen />);
    expect(screen.getByText(/no rows to display/i)).toBeInTheDocument();
  });

  it('triggers a CSV download when the download button is clicked', () => {
    const createObjectURL = vi.fn(() => 'blob:mock');
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;

    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    render(<ChartDataTable columns={columns} rows={rows} filename="my-export" caption="Test data" />);
    fireEvent.click(screen.getByRole('button', { name: /download csv/i }));

    expect(createObjectURL).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();

    clickSpy.mockRestore();
  });
});
