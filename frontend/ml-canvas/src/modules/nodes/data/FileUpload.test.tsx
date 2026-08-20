import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { FileUpload } from './FileUpload';
import type { UploadConfig } from '../../../core/api/client';

const mocks = vi.hoisted(() => ({
  uploadConfig: undefined as UploadConfig | undefined,
  uploadDataset: vi.fn(),
}));

vi.mock('../../../core/hooks/useUploadConfig', () => ({
  useUploadConfig: () => ({ data: mocks.uploadConfig }),
}));

vi.mock('../../../core/hooks/useDatasets', () => ({
  useUploadDataset: () => ({
    isPending: false,
    mutateAsync: mocks.uploadDataset,
  }),
}));

vi.mock('../../../core/toast', () => ({
  toast: {
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn(),
  },
}));

function renderUpload() {
  return render(
    <FileUpload onUploadComplete={() => {}} onCancel={() => {}} />,
  );
}

function makeFile(sizeBytes: number): File {
  const buffer = new ArrayBuffer(sizeBytes);
  return new File([buffer], 'data.csv', { type: 'text/csv' });
}

function dropFile(file: File) {
  const input = document.querySelector('input[type="file"]')!;
  const dropTarget = input.closest('label')!.parentElement!;
  fireEvent.drop(dropTarget, { dataTransfer: { files: [file] } });
}

describe('FileUpload', () => {
  beforeEach(() => {
    mocks.uploadConfig = undefined;
    mocks.uploadDataset.mockReset();
  });

  it('skips client-side checks before the config loads (server still enforces)', () => {
    mocks.uploadDataset.mockResolvedValue({ job_id: 'job-1' });
    renderUpload();

    const input = document.querySelector('input[type="file"]')!;
    expect(input.getAttribute('accept')).toBe('');
    expect(screen.queryByText(/Supports /)).not.toBeInTheDocument();

    // No size check before config loads — the drop proceeds to upload.
    const hugeFile = makeFile(1);
    Object.defineProperty(hugeFile, 'size', { value: 10 * 1024 * 1024 * 1024 + 1 });
    dropFile(hugeFile);
    expect(mocks.uploadDataset).toHaveBeenCalledTimes(1);
  });

  it('uses the backend upload config for the accept attribute and size limit', () => {
    mocks.uploadConfig = {
      max_upload_size_bytes: 100,
      allowed_extensions: ['.csv', '.json'],
    };

    renderUpload();

    const input = document.querySelector('input[type="file"]')!;
    expect(input.getAttribute('accept')).toBe('.csv,.json');
    expect(screen.getByText('Supports CSV, JSON')).toBeInTheDocument();

    dropFile(makeFile(200));
    expect(screen.getByText('File is too large (200 Bytes). Maximum size is 100 Bytes.')).toBeInTheDocument();
  });

  it('formats the 10GB default limit from the backend config', () => {
    mocks.uploadConfig = {
      max_upload_size_bytes: 10 * 1024 * 1024 * 1024,
      allowed_extensions: ['.csv'],
    };

    renderUpload();

    const hugeFile = makeFile(1);
    Object.defineProperty(hugeFile, 'size', { value: 10 * 1024 * 1024 * 1024 + 1 });
    dropFile(hugeFile);
    expect(screen.getByText(/File is too large \(10 GB\)\. Maximum size is 10 GB\./)).toBeInTheDocument();
  });
});
