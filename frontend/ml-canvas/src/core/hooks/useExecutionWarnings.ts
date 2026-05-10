/**
 * Watches the canvas execution result for newly-arrived per-node warnings
 * and routes them into the persistent `useNotificationsStore` buffer
 * (for the navbar bell), then auto-opens the bell so the user sees them.
 * Also persists all node failures + warnings to the backend DB so they
 * survive page refreshes and are visible on the /errors page Pipeline tab.
 *
 * Intentionally NO toast spam — the bell opening itself is the signal.
 *
 * Mounted once near the canvas root.
 */

import { useEffect, useRef } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { useNotificationsStore } from '../store/useNotificationsStore';
import { monitoringApi } from '../api/monitoring';
import type { PipelineLogEntry } from '../api/monitoring';

/** Dispatched to tell the notification bell to open itself. */
export const OPEN_BELL_EVENT = 'skyulf:open-bell';

export const useExecutionWarnings = (): void => {
  const executionResult = useGraphStore((s) => s.executionResult);
  const addMany = useNotificationsStore((s) => s.addMany);
  const seenRef = useRef<unknown>(null);

  useEffect(() => {
    if (!executionResult) return;
    if (seenRef.current === executionResult) return;
    seenRef.current = executionResult;

    const warnings = executionResult.node_warnings ?? [];

    // Feed warnings into the local notification store (bell icon + localStorage).
    if (warnings.length > 0) {
      addMany(warnings);
      window.dispatchEvent(new Event(OPEN_BELL_EVENT));
    }

    // Build a batch of entries to persist to the backend DB.
    const entries: PipelineLogEntry[] = [];

    // Node failures — status === 'failed' in node_results.
    const nodeResults = executionResult.node_results ?? {};
    for (const [nodeId, result] of Object.entries(nodeResults)) {
      if (result?.status === 'failed' && result.error) {
        entries.push({
          node_id: nodeId,
          node_type: null,
          level: 'error',
          logger: 'engine',
          message: result.error,
        });
      }
    }

    // Soft per-node warnings (TargetEncoder coercions, OHE degenerates, etc.)
    for (const w of warnings) {
      entries.push({
        node_id: w.node_id ?? null,
        node_type: w.node_type ?? null,
        level: w.level,
        logger: w.logger,
        message: w.message,
      });
    }

    if (entries.length > 0) {
      // Fire-and-forget — do not block the render or surface errors to user.
      monitoringApi
        .logPipelineRun(executionResult.pipeline_id ?? null, entries)
        .catch(() => { /* silently ignore if backend is down */ });
    }
  }, [executionResult, addMany]);
};
