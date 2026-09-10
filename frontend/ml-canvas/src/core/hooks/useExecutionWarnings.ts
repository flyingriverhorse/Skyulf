/**
 * Watches the canvas execution result for newly-arrived per-node warnings
 * and routes them into the persistent `useNotificationsStore` buffer
 * (for the navbar bell), then emits the existing bell event for listeners.
 * Also persists all node failures + warnings to the backend DB so they
 * survive page refreshes and are visible on the /errors page Pipeline tab.
 *
 * The notification buffer supplies the navbar badge without transient toasts.
 *
 * Mounted once near the canvas root.
 */

import { useEffect, useRef } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { useNotificationsStore } from '../store/useNotificationsStore';
import { monitoringApi } from '../api/monitoring';
import type { PipelineLogEntry } from '../api/monitoring';
import type { PreviewResponse, NodeWarning } from '../api/client';

/** Dispatched to tell the notification bell to open itself. */
export const OPEN_BELL_EVENT = 'skyulf:open-bell';

/** Persist node failures in response order, followed by every soft warning. */
function buildPipelineLogEntries(executionResult: PreviewResponse, warnings: NodeWarning[]): PipelineLogEntry[] {
  const entries: PipelineLogEntry[] = [];
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
  for (const warning of warnings) entries.push(warningLogEntry(warning));
  return entries;
}

/** Normalize only absent warning identifiers, retaining empty strings. */
function warningLogEntry(warning: NodeWarning): PipelineLogEntry {
  return {
    node_id: warning.node_id ?? null,
    node_type: warning.node_type ?? null,
    level: warning.level,
    logger: warning.logger,
    message: warning.message,
  };
}

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

    const entries = buildPipelineLogEntries(executionResult, warnings);

    if (entries.length > 0) {
      // Fire-and-forget — do not block the render or surface errors to user.
      monitoringApi
        .logPipelineRun(executionResult.pipeline_id ?? null, entries)
        .catch(() => { /* silently ignore if backend is down */ });
    }
  }, [executionResult, addMany]);
};
