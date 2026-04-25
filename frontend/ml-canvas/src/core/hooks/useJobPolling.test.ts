import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useJobPolling, isTerminalStatus } from './useJobPolling';
import { jobsApi, JobInfo, JobStatus } from '../api/jobs';

// Build a minimally-typed JobInfo. We only inspect `status` in this hook,
// so the rest is set to defensible defaults that satisfy the type.
const makeJob = (id: string, status: JobStatus): JobInfo => ({
  job_id: id,
  pipeline_id: 'p1',
  node_id: 'n1',
  job_type: 'basic_training',
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
});
