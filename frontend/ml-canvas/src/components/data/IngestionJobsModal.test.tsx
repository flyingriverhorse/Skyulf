import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import type { Dataset } from '../../core/types/api';
import { IngestionJobsModal } from './IngestionJobsModal';

const mocks = vi.hoisted(() => ({
  cancelIngestion: vi.fn(),
  confirm: vi.fn(),
  onRetry: vi.fn(),
}));

vi.mock('../../core/hooks/useDatasets', () => ({
  useCancelIngestion: () => ({
    isPending: false,
    variables: null,
    mutateAsync: mocks.cancelIngestion,
  }),
}));

interface MockModalShellProps {
  isOpen: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  children: React.ReactNode;
}

function MockModalShell({ isOpen, title, children }: MockModalShellProps): React.ReactElement | null {
  if (!isOpen) return null;
  return (
    <section aria-label={typeof title === 'string' ? title : undefined}>
      {typeof title === 'string' ? <h2>{title}</h2> : title}
      {children}
    </section>
  );
}

interface MockVirtualListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
}

function MockVirtualList<T>({ items, renderItem }: MockVirtualListProps<T>): React.ReactElement {
  return <div>{items.map((item, index) => renderItem(item, index))}</div>;
}

vi.mock('../shared', () => ({
  ModalShell: MockModalShell,
  VirtualList: MockVirtualList,
  useConfirm: () => mocks.confirm,
}));

const datasets: Dataset[] = [
  {
    id: 'source-processing',
    name: 'Quarterly Sales',
    type: 'file',
    created_at: '2026-08-07T09:00:00.000Z',
    source_metadata: {
      ingestion_status: {
        status: 'processing',
        progress: 42,
      },
    },
  },
  {
    id: 'source-failed',
    name: 'Broken Import',
    type: 's3',
    created_at: '2026-08-06T09:00:00.000Z',
    source_metadata: {
      ingestion_status: {
        status: 'failed',
        error: 'S3 error: permission denied',
      },
    },
  },
  {
    id: 'source-complete',
    name: 'Historical Snapshot',
    type: 'file',
    created_at: '2026-08-05T09:00:00.000Z',
    source_metadata: {
      ingestion_status: {
        status: 'completed',
      },
    },
  },
];

const failedDataset = datasets[1]!;

describe('IngestionJobsModal', () => {
  beforeEach(() => {
    mocks.cancelIngestion.mockReset();
    mocks.confirm.mockReset();
    mocks.onRetry.mockReset();
    mocks.confirm.mockResolvedValue(true);
  });

  it('separates active ingestions from terminal history and surfaces failure controls', async () => {
    render(
      <IngestionJobsModal
        isOpen
        onClose={vi.fn()}
        datasets={datasets}
        onRetry={mocks.onRetry}
      />,
    );

    expect(screen.getByRole('heading', { name: 'Active ingestions' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Ingestion history' })).toBeInTheDocument();

    expect(within(screen.getByRole('region', { name: 'Active ingestions' })).getByText('Quarterly Sales')).toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'Ingestion history' })).getByText('Historical Snapshot')).toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'Ingestion history' })).getByText('Broken Import')).toBeInTheDocument();
    expect(screen.getByText('S3 error: permission denied')).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /cancel ingestion/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('keeps retry disabled while the retry flow is running', async () => {
    let resolveRetry: (() => void) | undefined;
    mocks.onRetry.mockReturnValue(
      new Promise<void>((resolve) => {
        resolveRetry = resolve;
      }),
    );

    render(
      <IngestionJobsModal
        isOpen
        onClose={vi.fn()}
        datasets={[failedDataset]}
        onRetry={mocks.onRetry}
      />,
    );

    const retryButton = screen.getByRole('button', { name: /retry/i });
    fireEvent.click(retryButton);
    expect(retryButton).toBeDisabled();
    expect(mocks.onRetry).toHaveBeenCalledWith(failedDataset);

    resolveRetry?.();
    await waitFor(() => expect(retryButton).not.toBeDisabled());
  });
});
