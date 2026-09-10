import { describe, it, expect, vi, afterEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useJobPolling, isTerminalStatus } from './useJobPolling';
import { jobsApi, JobInfo, JobStatus } from '../api/jobs';
import { jobEventsSocket, type JobEvent } from '../realtime/jobEventsSocket';

// Build a minimally-typed JobInfo. We only inspect `status` in this hook,
// so the rest is set to defensible defaults that satisfy the type.
const makeJob = (id: string, status: JobStatus): JobInfo => ({
  job_id: id,
  pipeline_id: 'p1',
  node_id: 'n1',
  job_type: 'training',
  status,
  start_time: null,
  end_time: null,
  error: null,
  result: null,
  created_at: '2026-01-01T00:00:00Z',
});

describe('isTerminalStatus', () => {
  it('returns true for completed/succeeded/failed/cancelled', () => {
    expect(isTerminalStatus('completed')).toBe(true);
    expect(isTerminalStatus('succeeded')).toBe(true);
    expect(isTerminalStatus('failed')).toBe(true);
    expect(isTerminalStatus('cancelled')).toBe(true);
  });

  it('returns false for in-flight states + null/undefined', () => {
    expect(isTerminalStatus('running')).toBe(false);
    expect(isTerminalStatus('queued')).toBe(false);
    expect(isTerminalStatus('pending')).toBe(false);
    expect(isTerminalStatus(undefined)).toBe(false);
    expect(isTerminalStatus(null)).toBe(false);
  });
});

describe('useJobPolling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns idle state for an empty job list and never calls the API', async () => {
    const spy = vi.spyOn(jobsApi, 'getJob');
    const { result } = renderHook(() => useJobPolling([]));
    expect(result.current.aggregateStatus).toBe('idle');
    expect(result.current.isPolling).toBe(false);
    expect(spy).not.toHaveBeenCalled();
  });

  it('aggregates "completed" once every job is terminal and stops polling', async () => {
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('a', 'completed'));

    const { result } = renderHook(() => useJobPolling(['a'], { intervalMs: 100 }));

    // Initial fetch is fired immediately on mount.
    await waitFor(() => {
      expect(result.current.aggregateStatus).toBe('completed');
    });
    // stopOnTerminal default → polling halts after the terminal snapshot.
    await waitFor(() => {
      expect(result.current.isPolling).toBe(false);
    });
  });

  it('aggregates "failed" if any single job failed', async () => {
    vi.spyOn(jobsApi, 'getJob').mockImplementation(async (id: string) =>
      id === 'bad' ? makeJob('bad', 'failed') : makeJob(id, 'completed'),
    );

    const { result } = renderHook(() => useJobPolling(['ok', 'bad'], { intervalMs: 100 }));

    await waitFor(() => {
      expect(result.current.aggregateStatus).toBe('failed');
    });
  });

  it('settles as failed when every tracked job is cancelled', async () => {
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('cancelled', 'cancelled'));

    const { result } = renderHook(() => useJobPolling(['cancelled'], { intervalMs: 100 }));

    await waitFor(() => {
      expect(result.current.aggregateStatus).toBe('failed');
    });
    await waitFor(() => {
      expect(result.current.isPolling).toBe(false);
    });
  });

  it('reports "running" while at least one job is still in flight', async () => {
    vi.spyOn(jobsApi, 'getJob').mockImplementation(async (id: string) =>
      id === 'slow' ? makeJob('slow', 'running') : makeJob(id, 'completed'),
    );

    const { result } = renderHook(() => useJobPolling(['fast', 'slow'], { intervalMs: 100 }));

    await waitFor(() => {
      expect(result.current.aggregateStatus).toBe('running');
    });
    expect(result.current.isPolling).toBe(true);
  });

  it('does not restart polling when parent re-allocates the same id list', async () => {
    const spy = vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('a', 'running'));

    // Long interval so the periodic tick can't fire during the test.
    const { rerender } = renderHook(({ ids }) => useJobPolling(ids, { intervalMs: 60_000 }), {
      initialProps: { ids: ['a'] as readonly string[] },
    });

    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(1);
    });

    // Parent re-renders with a brand new array reference but same contents.
    rerender({ ids: ['a'] });
    rerender({ ids: ['a'] });

    // No additional fetch should fire from the rerenders themselves —
    // the idsKey signature didn't change so the effect must not restart.
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('gives up on a persistently-failing job after MAX_CONSECUTIVE_FETCH_FAILURES instead of polling forever', async () => {
    vi.spyOn(jobsApi, 'getJob').mockRejectedValue(new Error('404 not found'));

    const { result } = renderHook(() => useJobPolling(['deleted'], { intervalMs: 5 }));

    // Aggregate settles to 'error' once the job has failed enough
    // consecutive times, and polling stops instead of retrying forever.
    await waitFor(() => {
      expect(result.current.aggregateStatus).toBe('error');
    });
    await waitFor(() => {
      expect(result.current.isPolling).toBe(false);
    });
  });
});

/** Socket invalidations must debounce, preserve HTTP snapshots, and stop after cleanup. */
it('refetches tracked socket events and switches between safety-net and disconnected intervals', async () => {
  vi.useFakeTimers();
  let onEvent!: (event: JobEvent) => void;
  let onStatus!: (connected: boolean) => void;
  const unsubscribe = vi.fn();
  const unsubscribeStatus = vi.fn();
  vi.spyOn(jobEventsSocket, 'subscribe').mockImplementation(callback => { onEvent = callback; return unsubscribe; });
  vi.spyOn(jobEventsSocket, 'onStatus').mockImplementation(callback => { onStatus = callback; return unsubscribeStatus; });
  const getJob = vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('a', 'running'));
  const { result, unmount } = renderHook(() => useJobPolling(['a'], { intervalMs: 1000, skipInitialFetch: true }));
  try {
    expect(getJob).not.toHaveBeenCalled();
    act(() => { onEvent({ event: 'status', job_id: 'other' }); onStatus(true); onStatus(true); });
    act(() => { onEvent({ event: 'progress', job_id: 'a', progress: 90 }); });
    await act(async () => { await vi.advanceTimersByTimeAsync(250); });
    expect(getJob).toHaveBeenCalledTimes(1);
    expect(result.current.jobs.a).toEqual(makeJob('a', 'running'));
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(getJob).toHaveBeenCalledTimes(1);
    act(() => { onStatus(false); });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(getJob).toHaveBeenCalledTimes(2);
    act(() => { onEvent({ event: 'status', job_id: 'a' }); });
    unmount();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(getJob).toHaveBeenCalledTimes(2);
    expect(unsubscribe).toHaveBeenCalledOnce();
    expect(unsubscribeStatus).toHaveBeenCalledOnce();
  } finally {
    unmount();
    vi.restoreAllMocks();
    vi.useRealTimers();
  }
});

/** A response for a previous target list must not replace the newly selected job. */
it('ignores late HTTP results after switching targets and clears an empty selection', async () => {
  vi.spyOn(jobEventsSocket, 'subscribe').mockReturnValue(() => {});
  vi.spyOn(jobEventsSocket, 'onStatus').mockReturnValue(() => {});
  let resolveA!: (job: JobInfo) => void;
  vi.spyOn(jobsApi, 'getJob').mockReturnValueOnce(new Promise(resolve => { resolveA = resolve; }))
    .mockResolvedValue(makeJob('b', 'completed'));
  const { result, rerender, unmount } = renderHook(({ ids }) => useJobPolling(ids), { initialProps: { ids: ['a'] } });
  try {
    rerender({ ids: ['b'] });
    await waitFor(() => expect(result.current.jobs).toEqual({ b: makeJob('b', 'completed') }));
    await act(async () => { resolveA(makeJob('a', 'failed')); });
    expect(result.current.aggregateStatus).toBe('completed');
    expect(result.current.jobs).toEqual({ b: makeJob('b', 'completed') });
    rerender({ ids: [] });
    expect(result.current).toEqual({ jobs: {}, aggregateStatus: 'idle', isPolling: false });
  } finally {
    unmount();
    vi.restoreAllMocks();
  }
});
