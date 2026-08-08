import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useJobStore } from './useJobStore';
import { jobsApi, JobInfo } from '../api/jobs';

vi.mock('../realtime/jobEventsSocket', () => ({
  jobEventsSocket: {
    subscribe: vi.fn(() => () => {}),
    onStatus: vi.fn(() => () => {}),
  },
}));

const makeJob = (id: string, overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: id,
  pipeline_id: 'p1',
  node_id: 'n1',
  job_type: 'training',
  status: 'failed',
  start_time: null,
  end_time: null,
  error: null,
  result: null,
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
});

describe('useJobStore retry/cancel (double-submit guarding)', () => {
  beforeEach(() => {
    useJobStore.setState({ jobs: [], pendingJobActions: {}, skip: 0, hasMore: true, isLoading: false });
    vi.restoreAllMocks();
  });

  afterEach(() => {
    useJobStore.getState().stopPolling();
  });

  it('retryJob calls the API, refreshes the job list, and clears the pending flag', async () => {
    vi.spyOn(jobsApi, 'retryJob').mockResolvedValue({ job_id: 'job-2', message: 'ok' });
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([makeJob('job-2', { status: 'running' })]);

    const newId = await useJobStore.getState().retryJob('job-1');

    expect(newId).toBe('job-2');
    expect(jobsApi.retryJob).toHaveBeenCalledWith('job-1');
    expect(useJobStore.getState().jobs.map(j => j.job_id)).toEqual(['job-2']);
    expect(useJobStore.getState().pendingJobActions['job-1']).toBeUndefined();
  });

  it('refuses a second retryJob call for the same job while one is in flight', async () => {
    let resolveRetry: (() => void) | undefined;
    vi.spyOn(jobsApi, 'retryJob').mockImplementation(
      () =>
        new Promise(resolve => {
          resolveRetry = () => { resolve({ job_id: 'job-2', message: 'ok' }); };
        }),
    );
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([]);

    const first = useJobStore.getState().retryJob('job-1');
    // The pending flag must be set synchronously before the request settles.
    expect(useJobStore.getState().pendingJobActions['job-1']).toBe('retry');

    await expect(useJobStore.getState().retryJob('job-1')).rejects.toThrow(
      'An action is already in progress for this job',
    );
    expect(jobsApi.retryJob).toHaveBeenCalledTimes(1);

    resolveRetry?.();
    await first;
    expect(useJobStore.getState().pendingJobActions['job-1']).toBeUndefined();
  });

  it('clears the pending flag even when retryJob fails, so a later retry can be attempted', async () => {
    vi.spyOn(jobsApi, 'retryJob').mockRejectedValue(new Error('boom'));

    await expect(useJobStore.getState().retryJob('job-1')).rejects.toThrow('boom');
    expect(useJobStore.getState().pendingJobActions['job-1']).toBeUndefined();
  });

  it('cancelJob and retryJob guard each other: one in-flight action blocks the other for the same job', async () => {
    let resolveCancel: (() => void) | undefined;
    vi.spyOn(jobsApi, 'cancelJob').mockImplementation(
      () =>
        new Promise(resolve => {
          resolveCancel = () => { resolve(undefined); };
        }),
    );
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([]);

    const cancelPromise = useJobStore.getState().cancelJob('job-1');
    expect(useJobStore.getState().pendingJobActions['job-1']).toBe('cancel');

    await expect(useJobStore.getState().retryJob('job-1')).rejects.toThrow(
      'An action is already in progress for this job',
    );

    resolveCancel?.();
    await cancelPromise;
    expect(useJobStore.getState().pendingJobActions['job-1']).toBeUndefined();
  });

  it('does not block retrying a different job while one job has a pending action', async () => {
    let resolveRetryA: (() => void) | undefined;
    vi.spyOn(jobsApi, 'retryJob').mockImplementation((jobId: string) => {
      if (jobId === 'job-a') {
        return new Promise(resolve => {
          resolveRetryA = () => { resolve({ job_id: 'job-a-2', message: 'ok' }); };
        });
      }
      return Promise.resolve({ job_id: 'job-b-2', message: 'ok' });
    });
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([]);

    const pendingA = useJobStore.getState().retryJob('job-a');
    const newIdB = await useJobStore.getState().retryJob('job-b');

    expect(newIdB).toBe('job-b-2');
    resolveRetryA?.();
    await pendingA;
  });
});
