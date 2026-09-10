import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConfirmProvider } from '../components/shared';
import { DatasetService } from '../core/api/datasets';
import * as apiClient from '../core/api/client';
import { datasetKeys } from '../core/hooks/useDatasets';
import type { Dataset } from '../core/types/api';
import { DataSources } from './DataSources';

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });
const complete: Dataset = { id: 'ready', source_id: 'source-ready', name: 'Ready data', type: 'file', created_at: '2026-01-01', size_bytes: 0 };
const failed: Dataset = { id: 'failed', name: 'Remote data', type: 's3', format: 'parquet', created_at: '2026-01-02', source_metadata: { ingestion_status: { status: 'failed', error: 'Access denied' } } };

function LocationProbe() {
  /** Read the real router destination produced by dataset actions. */
  const location = useLocation();
  return <output aria-label="Current location">{location.pathname}{location.search}</output>;
}

function mountData(datasets = [complete, failed]) {
  /** Keep queries, modals, filtering and confirmation state real. */
  vi.spyOn(DatasetService, 'getAll').mockResolvedValue(datasets);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const result = render(<MemoryRouter initialEntries={['/data']}><QueryClientProvider client={client}><ConfirmProvider><DataSources /><LocationProbe /></ConfirmProvider></QueryClientProvider></MemoryRouter>);
  return { ...result, client };
}

describe('DataSources public lifecycle', () => {
  it('polls active ingestion and releases polling when the page unmounts', async () => {
    /** Navigating away must release the active ingestion query observer. */
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const pending: Dataset = { ...complete, source_metadata: { ingestion_status: { status: 'processing' } } };
    const { unmount } = mountData([pending]);
    await screen.findByText('Processing ingestion');
    const initialCalls = vi.mocked(DatasetService.getAll).mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(DatasetService.getAll).toHaveBeenCalledTimes(initialCalls + 1);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(DatasetService.getAll).toHaveBeenCalledTimes(initialCalls + 1);
  });

  it('validates upload size and closes the form after the ingestion response', async () => {
    /** The upload consumer must retain validation, file identity and completion behavior. */
    vi.spyOn(apiClient, 'fetchUploadConfig').mockResolvedValue({ max_upload_size_bytes: 3, allowed_extensions: ['.csv'] });
    const upload = vi.spyOn(DatasetService, 'uploadWithProgress').mockResolvedValue({ job_id: 'uploaded', status: 'pending' } as never);
    mountData([complete]);
    await screen.findByText('Ready data');
    fireEvent.click(screen.getByRole('button', { name: 'Upload File' }));
    await waitFor(() => expect(screen.getByLabelText('Browse dataset file')).toHaveAttribute('accept', '.csv'));
    fireEvent.change(screen.getByLabelText('Browse dataset file'), { target: { files: [new File(['1234'], 'large.csv')] } });
    expect(screen.getByText('File is too large (4 Bytes). Maximum size is 3 Bytes.')).toBeInTheDocument();
    expect(upload).not.toHaveBeenCalled();
    const file = new File(['x'], 'tiny.csv');
    fireEvent.change(screen.getByLabelText('Browse dataset file'), { target: { files: [file] } });
    await waitFor(() => expect(screen.queryByLabelText('Browse dataset file')).not.toBeInTheDocument());
    expect(upload).toHaveBeenCalledWith(file, expect.any(Function));
    expect(screen.getByRole('button', { name: 'Upload File' })).toBeInTheDocument();
  });

  it('preserves format and size fallbacks and keeps filters across refreshed query data', async () => {
    /** Table values and user filtering must survive server-cache refreshes. */
    const { client } = mountData();
    const row = (await screen.findByText('Ready data')).closest('tr')!;
    expect(within(row).getAllByRole('cell')[2]).toHaveTextContent('CSV');
    expect(within(row).getAllByRole('cell')[3]).toHaveTextContent('0 Bytes');
    expect(within(row).getByTitle('Dataset ID')).toHaveTextContent('source-ready');
    fireEvent.change(screen.getByRole('textbox', { name: 'Search datasets' }), { target: { value: 'SOURCE-READY' } });
    expect(screen.queryByText('Remote data')).not.toBeInTheDocument();
    client.setQueryData(datasetKeys.list('all'), [complete, failed, { ...complete, id: 'new', source_id: 'other', name: 'Other data' }]);
    await waitFor(() => expect(screen.queryByText('Other data')).not.toBeInTheDocument());
    expect(screen.getByRole('textbox', { name: 'Search datasets' })).toHaveValue('SOURCE-READY');
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
    expect(await screen.findByText('Remote data')).toBeInTheDocument();
  });

  it('uses dataset ids for canvas navigation and keeps row export pending until completion', async () => {
    /** Displayed source aliases must never replace dataset ids in actions. */
    let finish!: () => void;
    const exportData = vi.spyOn(DatasetService, 'exportData').mockReturnValue(new Promise<void>(resolve => { finish = resolve; }));
    mountData([complete]);
    await screen.findByText('Ready data');
    fireEvent.click(screen.getByTitle('Download CSV'));
    expect(exportData).toHaveBeenCalledWith('ready', 'csv');
    expect(screen.getByTitle('Download CSV')).toBeDisabled();
    finish();
    await waitFor(() => expect(screen.getByTitle('Download CSV')).not.toBeDisabled());
    fireEvent.click(screen.getByTitle('Use in Canvas'));
    expect(screen.getByLabelText('Current location')).toHaveTextContent('/canvas?source_id=ready');
  });

  it('requires delete confirmation and retries remote ingestion through the S3 form', async () => {
    /** Cancelled confirmation must not mutate the backend or hide the retry context. */
    const remove = vi.spyOn(DatasetService, 'delete').mockResolvedValue(undefined);
    mountData([failed]);
    await screen.findByText('Remote data');
    expect(screen.queryByRole('button', { name: 'Preview dataset' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Delete dataset' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Delete dataset?' })).getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(remove).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Retry ingestion' }));
    expect(await screen.findByLabelText('S3 Path')).toHaveValue('');
    expect(screen.getByRole('dialog', { name: 'Add Data Source' })).toBeInTheDocument();
  });
});
