import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { jobsApi, type JobInfo, type JobStatus } from '../api/jobs';
import { jobEventsSocket, type JobEvent } from '../realtime/jobEventsSocket';
import { useJobPolling } from './useJobPolling';

/** Hold an API response so tests choose completion order explicitly. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

/** Provide complete job snapshots with independently chosen lifecycle states. */
function job(status: JobStatus, id = 'a'): JobInfo {
  return {
    job_id: id, pipeline_id: 'p1', node_id: 'n1', job_type: 'training', status,
    start_time: null, end_time: null, error: null, result: null,
    created_at: '2026-09-12T00:00:00Z',
  };
}

describe('job polling request generations', () => {
  let onEvent: (event: JobEvent) => void;
  let onStatus: (connected: boolean) => void;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(jobEventsSocket, 'subscribe').mockImplementation(callback => {
      onEvent = callback;
      return () => {};
    });
    vi.spyOn(jobEventsSocket, 'onStatus').mockImplementation(callback => {
      onStatus = callback;
      return () => {};
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('keeps the newer terminal snapshot when an older response finishes later', async () => {
    // Regressing the request guard strands a running job with polling stopped.
    const first = deferred<JobInfo>();
    const second = deferred<JobInfo>();
    vi.spyOn(jobsApi, 'getJob').mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => useJobPolling(['a'], { intervalMs: 1000 }));
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await act(async () => { second.resolve(job('completed')); });
    await act(async () => { first.resolve(job('running')); });
    expect(result.current).toEqual({ jobs: { a: job('completed') }, aggregateStatus: 'completed', isPolling: false });
  });

  it('stops queued socket refreshes and reconnect intervals after terminal success', async () => {
    // A completed job must remain stopped even when socket callbacks arrive later.
    const response = deferred<JobInfo>();
    const getJob = vi.spyOn(jobsApi, 'getJob').mockReturnValue(response.promise);
    const { result } = renderHook(() => useJobPolling(['a'], { intervalMs: 1000 }));
    act(() => { onEvent({ event: 'status', job_id: 'a' }); });
    await act(async () => { response.resolve(job('completed')); });
    act(() => { onStatus(true); onStatus(false); onEvent({ event: 'status', job_id: 'a' }); });
    await act(async () => { await vi.advanceTimersByTimeAsync(35_000); });
    expect(getJob).toHaveBeenCalledTimes(1);
    expect(result.current.isPolling).toBe(false);
  });

  it('does not count stale failures toward the active request retry limit', async () => {
    // Older errors cannot exhaust the retry budget after a newer successful poll.
    const oldRequests = Array.from({ length: 4 }, () => deferred<JobInfo>());
    const getJob = vi.spyOn(jobsApi, 'getJob');
    oldRequests.forEach(request => getJob.mockReturnValueOnce(request.promise));
    getJob.mockResolvedValueOnce(job('running')).mockRejectedValue(new Error('offline'));
    const { result } = renderHook(() => useJobPolling(['a'], { intervalMs: 1000 }));
    await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
    await act(async () => { oldRequests.forEach(request => request.reject(new Error('old failure'))); });
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(result.current.aggregateStatus).toBe('running');
    expect(result.current.isPolling).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(4000); });
    expect(result.current.aggregateStatus).toBe('error');
    expect(result.current.isPolling).toBe(false);
  });

  it('publishes responses when every fetch takes longer than the poll interval', async () => {
    // Continuously starting a newer request must not starve all completed snapshots.
    vi.spyOn(jobsApi, 'getJob').mockImplementation(() => new Promise(resolve => {
      setTimeout(() => resolve(job('running')), 2000);
    }));
    const { result } = renderHook(() => useJobPolling(['a'], { intervalMs: 1000 }));
    await act(async () => { await vi.advanceTimersByTimeAsync(5500); });
    expect(result.current.jobs).toEqual({ a: job('running') });
    expect(result.current.isPolling).toBe(true);
  });
});
