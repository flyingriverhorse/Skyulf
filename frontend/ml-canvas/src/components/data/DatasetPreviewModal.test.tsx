import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DatasetPreviewModal } from './DatasetPreviewModal';
import { DatasetApiError, DatasetService } from '../../core/api/datasets';
import type { Dataset } from '../../core/types/api';

vi.mock('../../core/api/datasets', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/datasets')>('../../core/api/datasets');
  return {
    ...actual,
    DatasetService: {
      getSample: vi.fn(),
      getProfile: vi.fn(),
    },
  };
});

const mockedDatasetService = vi.mocked(DatasetService);

const dataset: Dataset = {
  id: 'dataset-1',
  name: 'Broken Dataset',
  type: 'file',
  created_at: '2026-08-07T00:00:00.000Z',
  format: 'CSV',
};

const renderPreview = () =>
  render(<DatasetPreviewModal dataset={dataset} isOpen onClose={vi.fn()} />);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('DatasetPreviewModal', () => {
  it('keeps real zero metadata visible when the profile succeeds', async () => {
    mockedDatasetService.getSample.mockResolvedValueOnce([{ id: 1 }]);
    mockedDatasetService.getProfile.mockResolvedValueOnce({
      metrics: {
        row_count: 0,
        column_count: 0,
        missing_cells: 0,
        missing_percentage: 0,
      },
      columns: [],
    });

    renderPreview();

    expect(await screen.findByText('0 rows')).toBeInTheDocument();
    expect(screen.getByText('0 columns')).toBeInTheDocument();
    expect(screen.getByLabelText('Unknown')).toBeInTheDocument();
  });

  it('renders unknown metadata instead of fabricating zeros when profile data is missing', async () => {
    mockedDatasetService.getSample.mockResolvedValueOnce([{ value: 'ok' }]);
    mockedDatasetService.getProfile.mockRejectedValueOnce(new DatasetApiError('Invalid file path', 400));

    renderPreview();

    await screen.findByText('ok');

    expect(screen.getAllByLabelText('Unknown')).toHaveLength(3);
    expect(screen.getByText('Broken Dataset')).toBeInTheDocument();
  });

  it('surfaces the backend sample error message and keeps the profile tab available', async () => {
    mockedDatasetService.getSample.mockRejectedValueOnce(new DatasetApiError('Invalid file path', 400));
    mockedDatasetService.getProfile.mockResolvedValueOnce({
      metrics: {
        row_count: 12,
        column_count: 4,
        missing_cells: 0,
        missing_percentage: 0,
      },
      columns: [],
    });

    renderPreview();

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Sample preview for "Broken Dataset"');
    expect(alert).toHaveTextContent('Invalid file path');

    fireEvent.click(screen.getByRole('button', { name: 'Statistics' }));
    expect(screen.getByText('12 rows')).toBeInTheDocument();
    expect(screen.getByText('4 columns')).toBeInTheDocument();
  });

  it('describes a missing profile source as deleted and keeps the sample visible', async () => {
    mockedDatasetService.getSample
      .mockResolvedValueOnce([{ value: 'sample row' }])
      .mockResolvedValueOnce([{ value: 'sample row' }]);
    mockedDatasetService.getProfile
      .mockRejectedValueOnce(new DatasetApiError('Not found', 404))
      .mockResolvedValueOnce({
        metrics: {
          row_count: 12,
          column_count: 4,
          missing_cells: 0,
          missing_percentage: 0,
        },
        columns: [],
      });

    renderPreview();

    await screen.findByText('sample row');

    fireEvent.click(screen.getByRole('button', { name: 'Statistics' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Statistics for "Broken Dataset"');
    expect(alert).toHaveTextContent(/deleted|missing/i);
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    await screen.findByText('12 rows');
    expect(screen.getByRole('button', { name: 'Statistics' })).toHaveClass('border-blue-600');
  });

  it('retries both requests without losing the selected tab', async () => {
    mockedDatasetService.getSample
      .mockResolvedValueOnce([{ value: 'sample row' }])
      .mockResolvedValueOnce([{ value: 'sample row' }]);
    mockedDatasetService.getProfile
      .mockRejectedValueOnce(new DatasetApiError('Invalid file path', 400))
      .mockResolvedValueOnce({
        metrics: {
          row_count: 12,
          column_count: 4,
          missing_cells: 0,
          missing_percentage: 0,
        },
        columns: [],
      });

    renderPreview();

    await screen.findByText('sample row');
    fireEvent.click(screen.getByRole('button', { name: 'Statistics' }));
    fireEvent.click(screen.getByRole('button', { name: /retry/i }));

    await screen.findByText('12 rows');
    await screen.findByText('4 columns');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Statistics' })).toHaveClass('border-blue-600'));
    expect(screen.getByText('12 rows')).toBeInTheDocument();
    expect(mockedDatasetService.getSample).toHaveBeenCalledTimes(2);
    expect(mockedDatasetService.getProfile).toHaveBeenCalledTimes(2);
  });

  it('treats null metadata from the backend as unknown rather than crashing', async () => {
    mockedDatasetService.getSample.mockResolvedValueOnce([{ value: 'ok' }]);
    mockedDatasetService.getProfile.mockRejectedValueOnce(new DatasetApiError('Invalid file path', 400));

    render(
      <DatasetPreviewModal
        dataset={{ ...dataset, rows: null, columns: null, size_bytes: null } as unknown as Dataset}
        isOpen
        onClose={vi.fn()}
      />,
    );

    await screen.findByText('ok');

    expect(screen.getAllByLabelText('Unknown')).toHaveLength(3);
  });
});
