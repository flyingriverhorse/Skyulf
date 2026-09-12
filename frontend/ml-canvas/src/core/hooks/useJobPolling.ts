import { useEffect, useRef, useState } from 'react';
import { jobsApi, JobInfo, JobStatus } from '../api/jobs';
import { jobEventsSocket } from '../realtime/jobEventsSocket';

// Cadence used as a safety net while the WebSocket is healthy. A live
// connection will normally invalidate within ~250 ms of any backend
// change, so a sparse interval is fine — it only matters if the
// connection drops or we missed a publish.
const SAFETY_NET_INTERVAL_MS = 30_000;

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

// If a job's fetch fails this many times in a row (e.g. the job was
// deleted server-side, or the backend is persistently erroring for
// that id), stop retrying it instead of polling forever with no way
// to ever reach a terminal state. A single job that will never
// recover previously kept `allTerminal`/`allCompleted` false forever,
// so `stopOnTerminal` never fired and the interval ran indefinitely.
const MAX_CONSECUTIVE_FETCH_FAILURES = 5;

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
   *  - `failed` if any job failed or all jobs reached an unsuccessful terminal state
   *  - `running` if any is still queued/running/pending
   *  - `completed` only when every job is in a terminal success state
   *  - `idle` when there are no job IDs to poll
   *  - `error` if the most recent fetch threw
   */
  aggregateStatus: 'idle' | 'running' | 'completed' | 'failed' | 'error';
  /** True while polling is still scheduled (not yet stopped). */
  isPolling: boolean;
}

/** Summarize a fetch batch without retaining snapshots for jobs that failed to load. */
function summarizeJobs(
  ids: readonly string[],
  results: (JobInfo | null)[],
  failureCounts: Record<string, number>,
) {
  const jobs: Record<string, JobInfo> = {};
  let anyFailed = false;
  let anyGaveUp = false;
  let allTerminal = true;
  let allCompleted = true;
  for (let i = 0; i < ids.length; i += 1) {
    const result = results[i];
    const id = ids[i]!;
    if (!result) {
      // Persistent failures eventually let stopOnTerminal settle the batch.
      if ((failureCounts[id] ?? 0) >= MAX_CONSECUTIVE_FETCH_FAILURES) {
        anyGaveUp = true;
        allCompleted = false;
        continue;
      }
      allTerminal = false;
      allCompleted = false;
      continue;
    }
    jobs[id] = result;
    const status = result.status;
    if (status === 'failed') anyFailed = true;
    if (!isTerminalStatus(status)) {
      allTerminal = false;
      allCompleted = false;
    } else if (status === 'failed' || status === 'cancelled') {
      allCompleted = false;
    }
  }
  return { jobs, anyFailed, anyGaveUp, allTerminal, allCompleted };
}

/** Failed jobs take precedence over exhausted requests and remaining active jobs. */
function aggregateJobStatus(summary: ReturnType<typeof summarizeJobs>): UseJobPollingResult['aggregateStatus'] {
  if (summary.anyFailed) return 'failed';
  if (summary.anyGaveUp) return 'error';
  if (summary.allCompleted && summary.allTerminal) return 'completed';
  if (summary.allTerminal) return 'failed';
  return 'running';
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

    const ids = idsRef.current;
    const failureCounts: Record<string, number> = {};
    let requestGeneration = 0;
    let appliedGeneration = 0;
    let cancelled = false;
    let stopped = false;
    let interval: ReturnType<typeof setInterval> | null = null;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;

    // Compare completed requests so slow APIs can still publish while a newer
    // request is pending. A stopped or retired polling lifetime accepts none.
    const isObsolete = (generation: number): boolean =>
      cancelled || stopped || generation < appliedGeneration;

    /** Terminal results also retire queued refreshes and socket-driven restarts. */
    const stopPolling = (): void => {
      stopped = true;
      if (interval) clearInterval(interval);
      if (refreshTimer) clearTimeout(refreshTimer);
      interval = null;
      refreshTimer = null;
      setIsPolling(false);
    };

    const fetchAll = async (): Promise<void> => {
      if (cancelled || stopped) return;
      const generation = ++requestGeneration;
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
        if (isObsolete(generation)) return;
        appliedGeneration = generation;

        results.forEach((result, i) => {
          const id = ids[i]!;
          failureCounts[id] = result ? 0 : (failureCounts[id] ?? 0) + 1;
        });
        const summary = summarizeJobs(ids, results, failureCounts);
        setJobs(summary.jobs);
        setAggregateStatus(aggregateJobStatus(summary));

        if (stopOnTerminal && summary.allTerminal) stopPolling();
      } catch (err) {
        if (!isObsolete(generation)) {
          appliedGeneration = generation;
          console.error('useJobPolling: unexpected error', err);
          setAggregateStatus('error');
        }
      }
    };

    setJobs({});
    setAggregateStatus('running');
    setIsPolling(true);
    if (!skipInitialFetch) void fetchAll();

    // WebSocket-driven invalidation. Each event carries a job_id; if
    // it matches one we're tracking we trigger an immediate (debounced)
    // refetch so the UI converges in ~250 ms instead of waiting for
    // the next interval tick.
    let wsConnected = false;
    const scheduleRefresh = (): void => {
      if (cancelled || stopped || refreshTimer) return;
      refreshTimer = setTimeout(() => {
        refreshTimer = null;
        void fetchAll();
      }, 250);
    };

    const startInterval = (ms: number): void => {
      if (cancelled || stopped) return;
      if (interval) clearInterval(interval);
      interval = setInterval(() => { void fetchAll(); }, ms);
    };

    const unsubscribeWs = jobEventsSocket.subscribe((evt) => {
      if (ids.includes(evt.job_id)) scheduleRefresh();
    });
    const unsubscribeStatus = jobEventsSocket.onStatus((connected) => {
      if (connected === wsConnected) return;
      wsConnected = connected;
      // Stretch the safety net while WS is up; tighten back on drop.
      startInterval(connected ? SAFETY_NET_INTERVAL_MS : intervalMs);
      if (connected) scheduleRefresh();
    });

    startInterval(wsConnected ? SAFETY_NET_INTERVAL_MS : intervalMs);

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
      if (refreshTimer) clearTimeout(refreshTimer);
      unsubscribeWs();
      unsubscribeStatus();
      setIsPolling(false);
    };
    // idsKey is the stable signature; intervalMs/stopOnTerminal/skipInitialFetch
    // legitimately need a restart when changed.
  }, [idsKey, intervalMs, stopOnTerminal, skipInitialFetch]);

  return { jobs, aggregateStatus, isPolling };
}
