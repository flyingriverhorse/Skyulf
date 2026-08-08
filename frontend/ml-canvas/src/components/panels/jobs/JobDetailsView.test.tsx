import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { JobDetailsView } from './JobDetailsView';
import { jobsApi, JobInfo } from '../../../core/api/jobs';

const mocks = vi.hoisted(() => ({
  cancelJob: vi.fn(),
  retryJob: vi.fn(),
  confirm: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
}));

vi.mock('../../../core/store/useJobStore', () => ({
  useJobStore: () => ({ cancelJob: mocks.cancelJob, retryJob: mocks.retryJob }),
}));

vi.mock('../../../core/toast', () => ({
  toast: { success: mocks.toastSuccess, error: mocks.toastError },
}));

vi.mock('../../shared', async () => {
  const actual = await vi.importActual<typeof import('../../shared')>('../../shared');
  return { ...actual, useConfirm: () => mocks.confirm };
});

const makeJob = (overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: 'job-1234567890',
  pipeline_id: 'pipe-1',
  node_id: 'node-1',
  job_type: 'training',
  status: 'failed',
  start_time: '2026-01-01T00:00:00Z',
  end_time: '2026-01-01T00:05:00Z',
  error: null,
  result: null,
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
});

function renderDetails(job: JobInfo, props: Partial<React.ComponentProps<typeof JobDetailsView>> = {}) {
  const onBack = vi.fn();
  const onClose = vi.fn();
  render(
    <MemoryRouter>
      <JobDetailsView job={job} onBack={onBack} onClose={onClose} {...props} />
    </MemoryRouter>,
  );
  return { onBack, onClose };
}

describe('JobDetailsView', () => {
  let originalScrollIntoView: typeof HTMLElement.prototype.scrollIntoView | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom has no layout engine and doesn't implement scrollIntoView; the
    // Logs tab's auto-scroll effect calls it on every render.
    originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = vi.fn();
    // Non-terminal jobs feed their id into useJobPolling, which fetches
    // for real via jobsApi.getJob — stub it so in-flight jobs don't hit
    // the network (jsdom otherwise logs a noisy connection failure).
    vi.spyOn(jobsApi, 'getJob').mockImplementation((id: string) => Promise.resolve(makeJob({ job_id: id, status: 'running' })));
  });

  afterEach(() => {
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView ?? (() => {});
  });

  it('names the full job id for assistive tech even though the header truncates it', () => {
    renderDetails(makeJob({ job_id: 'job-abcdef123456' }));
    const idChip = screen.getByTitle('job-abcdef123456');
    // The full id is present in the DOM (sr-only), not just the 8-char prefix.
    expect(idChip).toHaveTextContent('job-abcdef123456');
  });

  it('shows an active Retry action for a failed, retryable training job', async () => {
    mocks.confirm.mockResolvedValue(true);
    mocks.retryJob.mockResolvedValue('job-new-1');
    const { onBack } = renderDetails(makeJob({ status: 'failed', job_type: 'training' }));

    const retryButton = screen.getByRole('button', { name: /^Retry$/ });
    expect(retryButton).toBeEnabled();

    await act(async () => {
      fireEvent.click(retryButton);
    });

    await waitFor(() => {
      expect(mocks.retryJob).toHaveBeenCalledWith('job-1234567890');
    });
    expect(onBack).toHaveBeenCalled();
  });

  it('never double-submits retry: a second click while the first is in flight is ignored', async () => {
    mocks.confirm.mockResolvedValue(true);
    let resolveRetry: (() => void) | undefined;
    mocks.retryJob.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveRetry = () => { resolve('job-new-1'); };
        }),
    );
    renderDetails(makeJob({ status: 'failed', job_type: 'training' }));

    const retryButton = screen.getByRole('button', { name: /^Retry$/ });
    await act(async () => {
      fireEvent.click(retryButton);
    });

    // Button now reads "Retrying..." and is disabled — a second click must not fire another call.
    const retryingButton = screen.getByRole('button', { name: /Retrying/ });
    expect(retryingButton).toBeDisabled();
    fireEvent.click(retryingButton);

    expect(mocks.retryJob).toHaveBeenCalledTimes(1);
    resolveRetry?.();
    await waitFor(() => expect(screen.getByRole('button', { name: /^Retry$/ })).toBeEnabled());
  });

  it('explains rather than hides retry when the job type does not support it', () => {
    renderDetails(makeJob({ status: 'failed', job_type: 'eda' }));
    expect(screen.queryByRole('button', { name: /^Retry$/ })).not.toBeInTheDocument();
    const unavailable = screen.getByText('Retry unavailable');
    expect(unavailable).toHaveAttribute('title', expect.stringContaining("isn't available for eda jobs"));
  });

  it('explains retry unavailability for a job that already succeeded', () => {
    renderDetails(makeJob({ status: 'completed', job_type: 'training' }));
    const unavailable = screen.getByText('Retry unavailable');
    expect(unavailable).toHaveAttribute('title', expect.stringContaining('completed successfully'));
  });

  it('shows a Stop action for a running job and guards it against double-submission', async () => {
    mocks.confirm.mockResolvedValue(true);
    let resolveCancel: (() => void) | undefined;
    mocks.cancelJob.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveCancel = () => { resolve(undefined); };
        }),
    );
    renderDetails(makeJob({ status: 'running' }));

    const stopButton = screen.getByRole('button', { name: /Stop Job/ });
    await act(async () => {
      fireEvent.click(stopButton);
    });

    const stoppingButton = screen.getByRole('button', { name: /Stopping/ });
    expect(stoppingButton).toBeDisabled();
    fireEvent.click(stoppingButton);
    expect(mocks.cancelJob).toHaveBeenCalledTimes(1);

    resolveCancel?.();
    await waitFor(() => expect(mocks.cancelJob).toHaveBeenCalledTimes(1));
    // Let the running job's background poll settle too, to keep the test quiet.
    await waitFor(() => { expect(jobsApi.getJob).toHaveBeenCalled(); });
  });

  it('shows "No logs available" rather than a blank panel when a job has no logs yet', async () => {
    renderDetails(makeJob({ status: 'running', logs: [] }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(await screen.findByText(/No logs available/i)).toBeInTheDocument();
    await waitFor(() => { expect(jobsApi.getJob).toHaveBeenCalled(); });
  });

  it('renders log lines when present', async () => {
    renderDetails(makeJob({ status: 'failed', logs: ['INFO: starting', 'ERROR: boom'] }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(await screen.findByText(/starting/)).toBeInTheDocument();
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it('shows the job error alongside logs and result context when the job failed', () => {
    renderDetails(makeJob({ status: 'failed', error: 'Something exploded' }));
    expect(screen.getByText('Something exploded')).toBeInTheDocument();
  });

  it('links related records (dataset, pipeline) using the shared RecordLink primitive', () => {
    renderDetails(
      makeJob({ dataset_id: 'ds-1', dataset_name: 'Sales Data', pipeline_id: 'pipe-42' }),
      { origin: '/jobs', filters: { tab: 'classification' } },
    );
    const datasetLink = screen.getByRole('link', { name: 'Dataset ds-1' });
    expect(datasetLink.getAttribute('href')).toContain('oc.kind=dataset');
    expect(datasetLink.getAttribute('href')).toContain('oc.origin=%2Fjobs');

    const pipelineLink = screen.getByRole('link', { name: 'Pipeline pipe-42' });
    expect(pipelineLink.getAttribute('href')).toContain('oc.kind=pipeline');
  });

  it('reports progress honestly instead of implying a bar it cannot back', async () => {
    renderDetails(makeJob({ status: 'running', end_time: null }));
    // Let the in-flight poll's promise settle before asserting, so React
    // doesn't warn about an unwrapped state update from the mocked fetch.
    await waitFor(() => { expect(jobsApi.getJob).toHaveBeenCalled(); });
    const progressLabel = screen.getByText('Progress');
    const progressValue = within(progressLabel.parentElement as HTMLElement).getByText('Not reported');
    expect(progressValue).toBeInTheDocument();
  });
});
