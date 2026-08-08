import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Dataset } from '../core/types/api';
import { DataSources } from './DataSources';

const mocks = vi.hoisted(() => ({
  cancelIngestion: vi.fn(),
  confirm: vi.fn(),
  deleteDataset: vi.fn(),
  navigate: vi.fn(),
  refetch: vi.fn(),
  failedDataset: {
    id: 'dataset-1',
    source_id: 'source-1',
    name: 'Broken CSV',
    type: 'file',
    format: 'csv',
    created_at: '2026-08-07T08:00:00.000Z',
    size_bytes: 128,
    source_metadata: {
      ingestion_status: {
        status: 'failed',
        error: 'Upload failed: invalid delimiter',
      },
    },
  } as Dataset,
}));

vi.mock('react-router-dom', () => ({
  useNavigate: () => mocks.navigate,
}));

vi.mock('../core/hooks/useDatasets', () => ({
  hasPendingIngestion: () => false,
  useDatasets: () => ({
    data: [mocks.failedDataset],
    isLoading: false,
    refetch: mocks.refetch,
  }),
  useDeleteDataset: () => ({
    isPending: false,
    variables: null,
    mutateAsync: mocks.deleteDataset,
  }),
  useCancelIngestion: () => ({
    isPending: false,
    variables: null,
    mutateAsync: mocks.cancelIngestion,
  }),
}));

vi.mock('../core/toast', () => ({
  toast: {
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock('../components/data/DatasetPreviewModal', () => ({
  DatasetPreviewModal: () => null,
}));

vi.mock('../components/data/AddSourceModal', () => ({
  AddSourceModal: () => null,
}));

vi.mock('../components/data/IngestionJobsModal', () => ({
  IngestionJobsModal: () => null,
}));

vi.mock('../components/data/PipelineVersionsModal', () => ({
  PipelineVersionsModal: () => null,
}));

vi.mock('../modules/nodes/data/FileUpload', () => ({
  FileUpload: () => <div>Upload Form</div>,
}));

vi.mock('../components/shared', () => ({
  LoadingState: ({ message }: { message?: string }) => <div>{message}</div>,
  EmptyState: ({ title }: { title: string }) => <div>{title}</div>,
  useConfirm: () => mocks.confirm,
}));

describe('DataSources', () => {
  beforeEach(() => {
    mocks.cancelIngestion.mockReset();
    mocks.confirm.mockReset();
    mocks.deleteDataset.mockReset();
    mocks.navigate.mockReset();
    mocks.refetch.mockReset();
    mocks.confirm.mockResolvedValue(true);
  });

  it('shows failed-job details inline and routes retry to the upload flow', async () => {
    render(<DataSources />);

    expect(screen.getByText('Upload failed: invalid delimiter')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    await waitFor(() => expect(screen.getByText('Upload Form')).toBeInTheDocument());
  });

  it('does not put persistent row failure text in an assertive live region', () => {
    render(<DataSources />);

    const message = screen.getByText('Upload failed: invalid delimiter');
    expect(message.closest('[role="alert"], [aria-live="assertive"]')).toBeNull();
  });
});
