import { useEffect } from 'react';

import { jobsApi } from '../api/jobs';
import { jobEventsSocket } from '../realtime/jobEventsSocket';
import { useGraphStore } from '../store/useGraphStore';

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
 * Cheap: a single small JSON dict per refresh; the WS event already
 * fires only on actual job state changes.
 */
export function useNodeJobSummaries(): void {
  const setNodeJobSummaries = useGraphStore((s) => s.setNodeJobSummaries);

  useEffect(() => {
    let cancelled = false;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;

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

    void fetchSummaries();
    const unsubscribe = jobEventsSocket.subscribe(() => {
      scheduleRefresh();
    });

    return () => {
      cancelled = true;
      if (refreshTimer) clearTimeout(refreshTimer);
      unsubscribe();
    };
  }, [setNodeJobSummaries]);
}
