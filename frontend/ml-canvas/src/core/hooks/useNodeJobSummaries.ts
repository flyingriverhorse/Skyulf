import { useEffect, useRef } from 'react';

import { jobsApi } from '../api/jobs';
import { jobEventsSocket } from '../realtime/jobEventsSocket';
import { useGraphStore } from '../store/useGraphStore';
import { useJobStore } from '../store/useJobStore';

/**
 * Keeps `nodeJobSummaries` in the graph store in sync with the
 * backend's per-node card-summary endpoint.
 *
 * Why: trainer/tuner nodes execute via Celery jobs, not the inline
 * `/preview` path. The pipeline engine still builds a
 * `metadata.summary` per node, but only `job.metrics` is persisted,
 * so the FE's `executionResult.node_results` never gets a trainer
 * entry. This hook fetches the summaries on canvas mount and again
 * after every job event (debounced) so trainer cards refresh as soon
 * as a run completes.
 *
 * Safety-net poll: only runs while there are active (fresh) jobs so
 * an idle canvas never hits this endpoint. The WS event stream is the
 * primary refresh trigger; the 30 s interval is the fallback for when
 * the WS drops during a real run.
 */
export function useNodeJobSummaries(): void {
  const setNodeJobSummaries = useGraphStore((s) => s.setNodeJobSummaries);
  const jobs = useJobStore((s) => s.jobs);

  // Stable ref so the effect closure always sees the latest job list
  // without needing to re-register the WS subscription on every change.
  const jobsRef = useRef(jobs);
  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  useEffect(() => {
    let cancelled = false;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;
    let safetyPoll: ReturnType<typeof setInterval> | null = null;

    const TWO_HOURS_MS = 2 * 60 * 60 * 1000;

    const hasActiveJobs = (): boolean => {
      const now = Date.now();
      return jobsRef.current.some(
        (j) =>
          (j.status === 'running' || j.status === 'queued') &&
          now - new Date(j.created_at).getTime() < TWO_HOURS_MS,
      );
    };

    const fetchSummaries = async (): Promise<void> => {
      try {
        const summaries = await jobsApi.getNodeSummaries();
        if (!cancelled) setNodeJobSummaries(summaries);
      } catch {
        // A transient API error must not break the canvas; the next
        // job event (or remount) will retry.
      }
    };

    const scheduleRefresh = (): void => {
      if (refreshTimer) return;
      refreshTimer = setTimeout(() => {
        refreshTimer = null;
        void fetchSummaries();
      }, 400);
    };

    // Manage the safety-net interval: start it only when there are
    // active jobs and clear it as soon as there are none.
    const syncSafetyPoll = (): void => {
      if (hasActiveJobs()) {
        if (!safetyPoll) {
          safetyPoll = setInterval(() => {
            if (!hasActiveJobs()) {
              if (safetyPoll) { clearInterval(safetyPoll); safetyPoll = null; }
              return;
            }
            void fetchSummaries();
          }, 30_000);
        }
      } else {
        if (safetyPoll) { clearInterval(safetyPoll); safetyPoll = null; }
      }
    };

    // On mount: fetch once, then decide if the safety net is needed.
    void fetchSummaries().then(() => syncSafetyPoll());

    const unsubscribe = jobEventsSocket.subscribe(() => {
      scheduleRefresh();
      syncSafetyPoll();
    });

    return () => {
      cancelled = true;
      if (refreshTimer) clearTimeout(refreshTimer);
      if (safetyPoll) clearInterval(safetyPoll);
      unsubscribe();
    };
  }, [setNodeJobSummaries]);
}
