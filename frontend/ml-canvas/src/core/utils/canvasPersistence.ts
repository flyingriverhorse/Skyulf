/**
 * Auto-save / restore for the canvas graph (nodes + edges) via
 * `localStorage`. Separate from the server-side `savePipeline` /
 * `fetchPipeline` flow — this is purely a client-side safety net so
 * users don't lose unsaved work to an accidental tab close or refresh.
 *
 * Schema is intentionally minimal: just the React Flow `nodes` / `edges`
 * arrays plus a `savedAt` ISO timestamp. The `version` field is bumped
 * if we ever change the persisted shape so old payloads are ignored
 * instead of crashing the load.
 */
import type { Edge, Node } from '@xyflow/react';

const LS_KEY = 'skyulf:canvas:autosave:v1';
const SCHEMA_VERSION = 1;

export interface CanvasSnapshot {
  version: number;
  savedAt: string;
  nodes: Node[];
  edges: Edge[];
}

/** Persist the current canvas to `localStorage`. Best-effort; swallows
 *  quota / serialization errors so an autosave failure never breaks the
 *  app. */
export function saveCanvasSnapshot(nodes: Node[], edges: Edge[]): void {
  try {
    const payload: CanvasSnapshot = {
      version: SCHEMA_VERSION,
      savedAt: new Date().toISOString(),
      nodes,
      edges,
    };
    window.localStorage.setItem(LS_KEY, JSON.stringify(payload));
  } catch {
    // Quota exceeded or storage disabled — silently ignore.
  }
}

/** Read the most recent snapshot. Returns `null` when nothing is
 *  stored, the payload is corrupt, or the schema version doesn't
 *  match. */
export function loadCanvasSnapshot(): CanvasSnapshot | null {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      (parsed as { version?: unknown }).version !== SCHEMA_VERSION ||
      !Array.isArray((parsed as { nodes?: unknown }).nodes) ||
      !Array.isArray((parsed as { edges?: unknown }).edges)
    ) {
      return null;
    }
    return parsed as CanvasSnapshot;
  } catch {
    return null;
  }
}

/** Drop the saved snapshot (e.g. after the user explicitly chose
 *  "Discard" or successfully restored the session). */
export function clearCanvasSnapshot(): void {
  try {
    window.localStorage.removeItem(LS_KEY);
  } catch {
    // ignore
  }
}
