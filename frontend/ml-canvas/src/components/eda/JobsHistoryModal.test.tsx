import type { ComponentProps } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ConfirmProvider } from '../shared/ConfirmDialog';
import { EDAService } from '../../core/api/eda';
import { JobsHistoryModal } from './JobsHistoryModal';

vi.mock('../../core/api/eda', () => ({ EDAService: { cancelJob: vi.fn() } }));
type Props = ComponentProps<typeof JobsHistoryModal>;
type Report = Props['history'][number];

function renderHistory(overrides: Partial<Props> = {}) {
  /** Use the actual dialog and mutation so selection cannot bypass their controls. */
  const props: Props = {
    isOpen: true, onClose: vi.fn(), datasetId: 8,
    history: [{ id: 1, status: 'COMPLETED' }, { id: 2, status: 'RUNNING' }],
    onSelect: vi.fn(), onFetchReport: vi.fn(), onRefresh: vi.fn(), ...overrides,
  };
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { mutations: { retry: false } } })}>
    <ConfirmProvider><JobsHistoryModal {...props} /></ConfirmProvider>
  </QueryClientProvider>);
  return props;
}

describe('JobsHistoryModal public report controls', () => {
  beforeEach(() => { vi.clearAllMocks(); });

  it('loads the full selected report and calls selection before closing', async () => {
    /** Loading a summary must fetch its report and preserve the returned object. */
    const report = { id: 1, status: 'COMPLETED', description: 'Full report' };
    const calls: string[] = [];
    const props = renderHistory({ onFetchReport: vi.fn().mockResolvedValue(report),
      onSelect: vi.fn(() => { calls.push('select'); }), onClose: vi.fn(() => { calls.push('close'); }) });
    fireEvent.click(screen.getByText('Analysis #1'));
    expect(await screen.findByText('Analysis #1 Details')).toBeInTheDocument();
    expect(props.onFetchReport).toHaveBeenCalledExactlyOnceWith(1);
    expect(screen.getByText('Variables Overview')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Load this Report' }));
    expect(props.onSelect).toHaveBeenCalledExactlyOnceWith(report);
    expect(calls).toEqual(['select', 'close']);
  });

  it('keeps the latest clicked report when an older response arrives last', async () => {
    /** The in-flight guard protects the report that the user most recently chose. */
    let resolveFirst!: (report: Report) => void;
    let resolveSecond!: (report: Report) => void;
    const fetch = vi.fn().mockReturnValueOnce(new Promise<Report>(resolve => { resolveFirst = resolve; }))
      .mockReturnValueOnce(new Promise<Report>(resolve => { resolveSecond = resolve; }));
    renderHistory({ onFetchReport: fetch });
    fireEvent.click(screen.getByText('Analysis #1'));
    fireEvent.click(screen.getByText('Analysis #2'));
    await act(async () => { resolveSecond({ id: 2 }); });
    await act(async () => { resolveFirst({ id: 1 }); });
    expect(screen.getByText('Analysis #2 Details')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Back to history' }));
    expect(screen.getByText('Analysis History')).toBeInTheDocument();
  });

  it('formats zero report totals, excluded columns and variable missingness', async () => {
    /** Report details must retain their numeric fallbacks and per-column warning colors. */
    renderHistory({ onFetchReport: vi.fn().mockResolvedValue({ id: 1, profile_data: {
      row_count: 0, column_count: 1, missing_cells_percentage: 0, excluded_columns: ['secret'],
      columns: { amount: { name: 'amount', dtype: 'float64', missing_count: 1, missing_percentage: 2.5 } },
    } }) });
    fireEvent.click(screen.getByText('Analysis #1'));
    await screen.findByText('Analysis #1 Details');
    expect(screen.getByText('Rows').parentElement?.nextElementSibling).toHaveTextContent('0');
    expect(screen.getByText('Missing Cells').parentElement?.nextElementSibling).toHaveTextContent('0.0%');
    expect(screen.getByText('secret')).toHaveClass('line-through');
    const row = within(screen.getByRole('table')).getAllByRole('row')[1]!;
    expect(row).toHaveTextContent('amount');
    expect(within(row).getByText('2.5%')).toHaveClass('text-amber-600');
    expect(within(row).getByText('-')).toBeInTheDocument();
  });

  it('cancels only after confirmation without selecting the containing row', async () => {
    /** Cancel is a separate action and must not also fetch the clicked row. */
    vi.mocked(EDAService.cancelJob).mockResolvedValue({ message: 'cancelled' });
    const props = renderHistory();
    fireEvent.click(screen.getByTitle('Cancel Analysis'));
    const confirmation = screen.getByRole('dialog', { name: 'Cancel analysis?' });
    fireEvent.click(within(confirmation).getByRole('button', { name: 'Cancel' }));
    expect(EDAService.cancelJob).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTitle('Cancel Analysis'));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel job' }));
    await waitFor(() => { expect(props.onRefresh).toHaveBeenCalledOnce(); });
    expect(EDAService.cancelJob).toHaveBeenCalledExactlyOnceWith(2);
    expect(props.onFetchReport).not.toHaveBeenCalled();
  });
});
