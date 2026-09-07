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
    useJobStore.setState({ jobs: [], pendingJobActions: {}, nodeSubmissions: {}, inspectedRun: null, runJobs: {}, activeParallelRun: null, isDrawerOpen: false, skip: 0, hasMore: true, isLoading: false });
    vi.restoreAllMocks();
  });

  afterEach(() => {
    useJobStore.getState().stopPolling();
  });

  it('loads a submitted job outside the first page without corrupting history pagination', async () => {
    // An older run's View jobs link must still resolve its exact jobs after newer runs fill the page.
    const recent = Array.from({ length: 50 }, (_, i) => makeJob(`recent-${i}`));
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue(recent);
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('older', { status: 'completed' }));
    useJobStore.getState().setInspectedRun({ label: 'Training', jobIds: ['older'] });
    await useJobStore.getState().fetchJobs();
    expect(jobsApi.getJob).toHaveBeenCalledWith('older');
    expect(useJobStore.getState().runJobs.older?.status).toBe('completed');
    expect(useJobStore.getState().jobs).toHaveLength(50);
    expect(useJobStore.getState().skip).toBe(0);
  });

  it('retains a retried job in the inspected run when returning from details', async () => {
    // Retrying from scoped history must not hide the replacement job behind the original receipt.
    vi.spyOn(jobsApi, 'retryJob').mockResolvedValue({ job_id: 'new', message: 'ok' });
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([makeJob('old'), makeJob('new', { status: 'queued' })]);
    useJobStore.getState().setInspectedRun({ label: 'Training', jobIds: ['old'] });
    await useJobStore.getState().retryJob('old');
    expect(useJobStore.getState().inspectedRun?.jobIds).toEqual(['old', 'new']);
  });

  it('refreshes an active snapshot after its original receipt is replaced', async () => {
    // Old active jobs must reach a terminal state instead of keeping polling alive indefinitely.
    useJobStore.setState({ runJobs: { old: makeJob('old', { status: 'running' }) }, inspectedRun: null });
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([makeJob('old', { status: 'completed' })]);
    await useJobStore.getState().fetchJobs();
    expect(useJobStore.getState().runJobs.old?.status).toBe('completed');
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
    expect(useJobStore.getState().pendingJobActions).toEqual({});
  });
});

describe('useJobStore submitted run snapshots', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useJobStore.getState().stopPolling();
    useJobStore.setState({
      jobs: [], pendingJobActions: {}, nodeSubmissions: {}, inspectedRun: null,
      runJobs: {}, activeParallelRun: null, isDrawerOpen: false,
      skip: 0, hasMore: true, isLoading: false,
    });
    vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([]);
  });

  afterEach(() => {
    useJobStore.getState().stopPolling();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('deduplicates jobs shared by node receipts, inspected runs, and parallel runs', async () => {
    // Shared receipts must request each missing job once without dropping other nodes' submissions.
    const state = useJobStore.getState();
    state.setNodeSubmission('preparing', { pending: true, message: 'Submitting', run: null });
    state.setNodeSubmission('trainer', {
      pending: false, message: 'Submitted', run: { label: 'Training', jobIds: ['shared', 'node-only'] },
    });
    state.setInspectedRun({ label: 'Training', jobIds: ['shared', 'inspected-only'] });
    state.setActiveParallelRun({ jobIds: ['shared', 'parallel-only'], startedAt: '2026-09-07T12:00:00Z' });
    vi.spyOn(jobsApi, 'getJob').mockImplementation(async id => makeJob(id, { status: 'queued' }));

    await state.fetchJobs();

    expect(jobsApi.getJob).toHaveBeenCalledTimes(4);
    expect(useJobStore.getState().runJobs).toMatchObject({
      shared: { status: 'queued' }, 'node-only': { status: 'queued' },
      'inspected-only': { status: 'queued' }, 'parallel-only': { status: 'queued' },
    });
    expect(useJobStore.getState().nodeSubmissions.preparing).toEqual({
      pending: true, message: 'Submitting', run: null,
    });
  });

  it.each(['completed', 'succeeded', 'failed', 'cancelled'] as const)(
    'keeps an off-page %s snapshot without requesting it again', async status => {
      // Finished jobs remain inspectable even if their detail endpoint is no longer available.
      const cached = makeJob('older', { status });
      useJobStore.setState({ runJobs: { older: cached }, inspectedRun: { label: 'Run', jobIds: ['older'] } });
      const getJob = vi.spyOn(jobsApi, 'getJob');

      await useJobStore.getState().fetchJobs();

      expect(getJob).not.toHaveBeenCalled();
      expect(useJobStore.getState().runJobs.older).toBe(cached);
    },
  );

  it('uses fresh history instead of a terminal cached snapshot', async () => {
    // Refresh must expose server-side corrections to cached results when the job returns to the page.
    useJobStore.setState({
      runJobs: { corrected: makeJob('corrected', { status: 'failed' }) },
      inspectedRun: { label: 'Run', jobIds: ['corrected'] },
    });
    vi.mocked(jobsApi.getJobs).mockResolvedValue([makeJob('corrected', { status: 'completed' })]);
    const getJob = vi.spyOn(jobsApi, 'getJob');

    await useJobStore.getState().fetchJobs();

    expect(getJob).not.toHaveBeenCalled();
    expect(useJobStore.getState().runJobs.corrected?.status).toBe('completed');
  });

  it('retains the last active snapshot on a failed lookup and recovers on refresh', async () => {
    // A transient detail failure must not erase an active run or prevent its later completion.
    const cached = makeJob('older', { status: 'running' });
    useJobStore.setState({ runJobs: { older: cached } });
    vi.spyOn(jobsApi, 'getJob').mockRejectedValueOnce(new Error('temporarily unavailable'))
      .mockResolvedValueOnce(makeJob('older', { status: 'completed' }));

    await useJobStore.getState().fetchJobs();
    expect(useJobStore.getState().runJobs.older).toBe(cached);
    expect(useJobStore.getState().isLoading).toBe(false);
    await useJobStore.getState().fetchJobs();

    expect(useJobStore.getState().runJobs.older?.status).toBe('completed');
  });

  it('ignores mismatched lookup IDs without erasing existing run metadata', async () => {
    // A response for another job must never be attached to the selected run's receipt.
    const cached = makeJob('expected', { status: 'pending' });
    useJobStore.setState({
      runJobs: { expected: cached }, inspectedRun: { label: 'Run', jobIds: ['expected', 'missing'] },
    });
    vi.spyOn(jobsApi, 'getJob').mockImplementation(async id => {
      if (id === 'missing') throw new Error('not indexed yet');
      return makeJob('unrelated', { status: 'completed' });
    });

    await useJobStore.getState().fetchJobs();

    expect(useJobStore.getState().runJobs).toEqual({ expected: cached });
  });

  it('does not attach a retry to an unrelated inspected run', async () => {
    // Retrying elsewhere must not change which jobs a notification receipt displays.
    useJobStore.getState().setInspectedRun({ label: 'Other run', jobIds: ['other'] });
    vi.spyOn(jobsApi, 'retryJob').mockResolvedValue({ job_id: 'replacement', message: 'ok' });
    vi.mocked(jobsApi.getJobs).mockResolvedValue([makeJob('other')]);

    await useJobStore.getState().retryJob('failed');

    expect(useJobStore.getState().inspectedRun).toEqual({ label: 'Other run', jobIds: ['other'] });
  });

  it('keeps one copy of a replacement already present in the inspected run', async () => {
    // Repeated retry responses must not create duplicate job rows in scoped history.
    useJobStore.getState().setInspectedRun({ label: 'Run', jobIds: ['old', 'replacement'] });
    vi.spyOn(jobsApi, 'retryJob').mockResolvedValue({ job_id: 'replacement', message: 'ok' });
    vi.mocked(jobsApi.getJobs).mockResolvedValue([makeJob('old'), makeJob('replacement')]);

    await useJobStore.getState().retryJob('old');

    expect(useJobStore.getState().inspectedRun?.jobIds).toEqual(['old', 'replacement']);
  });

  it('clears the inspected receipt on close and fetches unscoped history on reopen', async () => {
    // Closing a notification's job view must not scope a later opening of normal history.
    useJobStore.setState({ isDrawerOpen: true, inspectedRun: { label: 'Old run', jobIds: ['old'] } });
    useJobStore.getState().toggleDrawer();
    expect(useJobStore.getState().inspectedRun).toBeNull();
    useJobStore.getState().toggleDrawer(true);
    await Promise.resolve();

    expect(jobsApi.getJobs).toHaveBeenCalledWith(50, 0);
    expect(useJobStore.getState().isDrawerOpen).toBe(true);
    expect(useJobStore.getState().inspectedRun).toBeNull();
  });

  it('keeps polling an off-page pending run until its snapshot becomes terminal', async () => {
    // Newly accepted jobs must keep receiving updates even when the first history page is empty.
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-07T12:00:00Z'));
    useJobStore.setState({
      runJobs: { pending: makeJob('pending', { status: 'pending', created_at: new Date().toISOString() }) },
    });
    vi.spyOn(jobsApi, 'getJob')
      .mockResolvedValueOnce(makeJob('pending', { status: 'queued', created_at: new Date().toISOString() }))
      .mockResolvedValueOnce(makeJob('pending', { status: 'completed' }));

    useJobStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(3000);
    expect(useJobStore.getState().runJobs.pending?.status).toBe('queued');
    await vi.advanceTimersByTimeAsync(3000);
    expect(useJobStore.getState().runJobs.pending?.status).toBe('completed');
    await vi.advanceTimersByTimeAsync(9000);

    expect(jobsApi.getJobs).toHaveBeenCalledTimes(2);
  });

  it('marks a parallel run complete from terminal off-page snapshots', async () => {
    // Successful aliases, failures, and cancellations all finish a branch, even outside history.
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-07T12:00:00Z'));
    useJobStore.setState({
      activeParallelRun: { jobIds: ['success', 'failure', 'cancel'], startedAt: '2026-09-07T11:59:00Z' },
      runJobs: {
        success: makeJob('success', { status: 'succeeded' }),
        failure: makeJob('failure', { status: 'failed' }),
        cancel: makeJob('cancel', { status: 'cancelled' }),
      },
    });

    useJobStore.getState().startPolling();
    await vi.advanceTimersByTimeAsync(3000);

    expect(useJobStore.getState().activeParallelRun?.completedAt).toBe('2026-09-07T12:00:03.000Z');
  });
});
