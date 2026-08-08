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

/** Discriminates every reason `loadCanvasSnapshot` used to collapse into a
 *  bare `null`, so callers (CAN-003) can explain *why* an autosave can't be
 *  restored instead of silently doing nothing:
 *  - `empty`: nothing has ever been saved.
 *  - `corrupt`: the payload isn't valid JSON, or is missing `nodes`/`edges`.
 *  - `version-mismatch`: valid JSON from an older/newer, incompatible schema.
 *  - `storage-error`: `localStorage` itself threw (quota exceeded, disabled,
 *    or unavailable in this browsing context). */
export type CanvasSnapshotDiagnostic =
  | { status: 'available'; snapshot: CanvasSnapshot }
  | { status: 'empty' }
  | { status: 'corrupt' }
  | { status: 'version-mismatch'; foundVersion: number | undefined }
  | { status: 'storage-error' };

/** Read the most recent snapshot along with a diagnostic explaining why it
 *  can't be restored when it can't be. Prefer this over `loadCanvasSnapshot`
 *  whenever the caller surfaces the result to the user (see
 *  `core/utils/canvasRecovery.ts`). */
export function loadCanvasSnapshotDiagnostic(): CanvasSnapshotDiagnostic {
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(LS_KEY);
  } catch {
    return { status: 'storage-error' };
  }
  if (!raw) return { status: 'empty' };

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { status: 'corrupt' };
  }
  if (!parsed || typeof parsed !== 'object') return { status: 'corrupt' };

  const nodes = (parsed as { nodes?: unknown }).nodes;
  const edges = (parsed as { edges?: unknown }).edges;
  if (!Array.isArray(nodes) || !Array.isArray(edges)) return { status: 'corrupt' };

  const version = (parsed as { version?: unknown }).version;
  if (version !== SCHEMA_VERSION) {
    return { status: 'version-mismatch', foundVersion: typeof version === 'number' ? version : undefined };
  }

  return { status: 'available', snapshot: parsed as CanvasSnapshot };
}

/** Read the most recent snapshot. Returns `null` when nothing is
 *  stored, the payload is corrupt, or the schema version doesn't
 *  match. Kept for callers that don't need to distinguish *why* — see
 *  `loadCanvasSnapshotDiagnostic` for the explainable variant. */
export function loadCanvasSnapshot(): CanvasSnapshot | null {
  const diagnostic = loadCanvasSnapshotDiagnostic();
  return diagnostic.status === 'available' ? diagnostic.snapshot : null;
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
