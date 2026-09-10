import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DatasetService } from '../../core/api/datasets';
import { AddSourceModal } from './AddSourceModal';

afterEach(() => vi.restoreAllMocks());

function mountSource() {
  /** Real query state pins pending submissions and callback ordering. */
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const result = render(<QueryClientProvider client={client}><AddSourceModal isOpen onClose={onClose} onSuccess={onSuccess} /></QueryClientProvider>);
  return { ...result, onClose, onSuccess };
}

describe('AddSourceModal public form', () => {
  it('validates only after submission and preserves untrimmed source values without incomplete credentials', async () => {
    /** Required-field feedback must not change the payload accepted by ingestion. */
    const create = vi.spyOn(DatasetService, 'createSource').mockResolvedValue({ job_id: 'job-1' } as never);
    const { onSuccess, onClose } = mountSource();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Create Source' }));
    expect(screen.getAllByRole('alert')).toHaveLength(2);
    expect(create).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: ' raw name ' } });
    fireEvent.change(screen.getByLabelText('S3 Path'), { target: { value: ' s3://bucket/data ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Credentials (Optional)' }));
    fireEvent.change(screen.getByLabelText('Access Key ID'), { target: { value: 'partial' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create Source' }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
    expect(create).toHaveBeenCalledWith({ name: ' raw name ', type: 's3', config: { path: ' s3://bucket/data ' }, description: 'Imported from s3' });
    expect(onSuccess).toHaveBeenCalledWith('job-1');
    expect(onSuccess.mock.invocationCallOrder[0]).toBeLessThan(onClose.mock.invocationCallOrder[0]!);
  });

  it('retains credentials through failure and disables submission until the response settles', async () => {
    /** Retrying a failed source must retain the exact optional storage configuration. */
    let rejectRequest!: (error: Error) => void;
    const create = vi.spyOn(DatasetService, 'createSource').mockReturnValue(new Promise((_, reject) => { rejectRequest = reject; }));
    const { onClose } = mountSource();
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Private' } });
    fireEvent.change(screen.getByLabelText('S3 Path'), { target: { value: 's3://private/data' } });
    fireEvent.click(screen.getByRole('button', { name: 'Credentials (Optional)' }));
    fireEvent.change(screen.getByLabelText('Access Key ID'), { target: { value: 'key' } });
    fireEvent.change(screen.getByLabelText('Secret Access Key'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create Source' }));
    expect(await screen.findByRole('button', { name: 'Creating...' })).toBeDisabled();
    expect(create).toHaveBeenCalledWith({ name: 'Private', type: 's3', config: { path: 's3://private/data', storage_options: { aws_access_key_id: 'key', aws_secret_access_key: 'secret', region: undefined } }, description: 'Imported from s3' });
    rejectRequest(new Error('Denied'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Denied');
    expect(screen.getByLabelText('Secret Access Key')).toHaveValue('secret');
    expect(onClose).not.toHaveBeenCalled();
  });
});
