import { useEffect, useRef, useState } from 'react';
import { jobsApi, JobInfo, JobStatus } from '../api/jobs';

// A status is "terminal" when no further state changes are expected
// from the backend. We stop polling once every tracked job has hit a
// terminal state — this is the single source of truth so callers
// don't each maintain their own list.
const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set<JobStatus>([
  'completed',
  'succeeded',
  'failed',
  'cancelled',
]);

export function isTerminalStatus(status: JobStatus | string | undefined | null): boolean {
  return !!status && TERMINAL_STATUSES.has(status as JobStatus);
}

export interface UseJobPollingOptions {
  /** Poll cadence in ms. Defaults to 2000. */
  intervalMs?: number;
  /**
   * If true (default), polling stops automatically once every job in
   * `jobIds` has reached a terminal status. Set false to keep
   * refreshing forever (e.g. an inspector that wants live timestamps
   * even after completion).
   */
  stopOnTerminal?: boolean;
  /** Skip the immediate fetch on mount/jobIds change. Defaults to false. */
  skipInitialFetch?: boolean;
}

export interface UseJobPollingResult {
  /** Latest snapshot keyed by job_id. Empty until the first fetch resolves. */
  jobs: Record<string, JobInfo>;
  /**
   * Aggregate status across all polled jobs:
   *  - `failed` if any single job failed
   *  - `running` if any is still queued/running/pending
   *  - `completed` only when every job is in a terminal success state
   *  - `idle` when there are no job IDs to poll
   *  - `error` if the most recent fetch threw
   */
  aggregateStatus: 'idle' | 'running' | 'completed' | 'failed' | 'error';
  /** True while polling is still scheduled (not yet stopped). */
  isPolling: boolean;
}

/**
 * Polls one-or-many backend jobs at a fixed cadence and exposes the
 * latest snapshots plus an aggregate status. Centralizes the
 * setInterval / cleanup / cancellation-guard pattern that previously
 * lived inline in DataPreviewComponents and JobsDrawer.
 *
 * Designed to be safe to call with an empty array — the hook simply
 * no-ops, which lets callers conditionally feed it `lastRunJobIds`
 * without an extra `if` wrapper.
 */
export function useJobPolling(
  jobIds: readonly string[],
  options: UseJobPollingOptions = {},
): UseJobPollingResult {
  const { intervalMs = 2000, stopOnTerminal = true, skipInitialFetch = false } = options;
  const [jobs, setJobs] = useState<Record<string, JobInfo>>({});
  const [aggregateStatus, setAggregateStatus] = useState<UseJobPollingResult['aggregateStatus']>(
    'idle',
  );
  const [isPolling, setIsPolling] = useState<boolean>(false);

  // We freeze the id list into a stable join-key so that a parent
  // re-rendering with a freshly-allocated `jobIds` array (same
  // contents) doesn't restart the interval on every render.
  const idsKey = jobIds.join('|');
  // Keep the latest ids reachable from inside the polling closure
  // without forcing the effect to depend on the array reference.
  const idsRef = useRef<readonly string[]>(jobIds);
  idsRef.current = jobIds;

  useEffect(() => {
    if (idsRef.current.length === 0) {
      setJobs({});
      setAggregateStatus('idle');
      setIsPolling(false);
      return undefined;
    }

    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const fetchAll = async (): Promise<void> => {
      const ids = idsRef.current;
      try {
        const results = await Promise.all(
          ids.map(async (id) => {
            try {
              return await jobsApi.getJob(id);
            } catch (err) {
              console.error('useJobPolling: fetch failed', id, err);
              return null;
            }
          }),
        );
        if (cancelled) return;

        const next: Record<string, JobInfo> = {};
        let anyFailed = false;
        let allTerminal = true;
        let allCompleted = true;
        for (let i = 0; i < ids.length; i += 1) {
          const result = results[i];
          const id = ids[i]!;
          if (!result) {
            // Treat fetch failures as non-terminal so we keep retrying.
            allTerminal = false;
            allCompleted = false;
            continue;
          }
          next[id] = result;
          const status = result.status;
          if (status === 'failed') anyFailed = true;
          if (!isTerminalStatus(status)) {
            allTerminal = false;
            allCompleted = false;
          } else if (status === 'failed' || status === 'cancelled') {
            allCompleted = false;
          }
        }
        setJobs(next);
        if (anyFailed) setAggregateStatus('failed');
        else if (allCompleted && allTerminal) setAggregateStatus('completed');
        else setAggregateStatus('running');

        if (stopOnTerminal && allTerminal && interval) {
          clearInterval(interval);
          interval = null;
          setIsPolling(false);
        }
      } catch (err) {
        if (!cancelled) {
          console.error('useJobPolling: unexpected error', err);
          setAggregateStatus('error');
        }
      }
    };

    setIsPolling(true);
    if (!skipInitialFetch) void fetchAll();
    interval = setInterval(() => {
      void fetchAll();
    }, intervalMs);

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
      setIsPolling(false);
    };
    // idsKey is the stable signature; intervalMs/stopOnTerminal/skipInitialFetch
    // legitimately need a restart when changed.
  }, [idsKey, intervalMs, stopOnTerminal, skipInitialFetch]);

  return { jobs, aggregateStatus, isPolling };
}
