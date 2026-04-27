/**
 * Ring buffer of recently saved pipelines, persisted in `localStorage`.
 *
 * Complements `canvasPersistence.ts` (single-slot autosave) by keeping a
 * named history the user can scroll back through from the Toolbar.
 *
 * Schema is intentionally tiny so quota is rarely a concern; entries are
 * dropped FIFO once the buffer is full. All writes are best-effort —
 * a quota or serialization error must never break Save.
 */
import type { Edge, Node } from '@xyflow/react';

const LS_KEY = 'skyulf:canvas:recent:v1';
const SCHEMA_VERSION = 1;
const MAX_ENTRIES = 5;

export interface RecentPipelineEntry {
  /** Stable id used as React key and to dedupe same-name re-saves. */
  id: string;
  /** User-facing label (today: pipeline name from Save dialog). */
  name: string;
  /** ISO timestamp; rendered as relative time in the UI. */
  savedAt: string;
  /** Optional: dataset id this pipeline targeted, so Restore can warn
   *  when applied to a different dataset. */
  datasetId?: string;
  /** Optional: human-readable dataset label captured at save time, so
   *  the dropdown can show e.g. "iris.csv" instead of an opaque id. */
  datasetName?: string;
  nodes: Node[];
  edges: Edge[];
}

interface StoredPayload {
  version: number;
  entries: RecentPipelineEntry[];
}

function readRaw(): StoredPayload | null {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      (parsed as { version?: unknown }).version !== SCHEMA_VERSION ||
      !Array.isArray((parsed as { entries?: unknown }).entries)
    ) {
      return null;
    }
    return parsed as StoredPayload;
  } catch {
    return null;
  }
}

/** Read the current ring buffer (most recent first). Returns `[]` when
 *  nothing is stored or the payload is corrupt. */
export function getRecentPipelines(): RecentPipelineEntry[] {
  return readRaw()?.entries ?? [];
}

/** Push a new entry. Same-name entries are deduped (newest wins) so the
 *  buffer doesn't fill up with re-saves of the same pipeline. Returns
 *  the resulting list (most recent first). */
export function pushRecentPipeline(
  entry: Omit<RecentPipelineEntry, 'id' | 'savedAt'> & { savedAt?: string },
): RecentPipelineEntry[] {
  const existing = getRecentPipelines().filter(
    (e) => e.name.trim().toLowerCase() !== entry.name.trim().toLowerCase(),
  );
  const next: RecentPipelineEntry = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    name: entry.name,
    savedAt: entry.savedAt ?? new Date().toISOString(),
    ...(entry.datasetId !== undefined ? { datasetId: entry.datasetId } : {}),
    ...(entry.datasetName !== undefined ? { datasetName: entry.datasetName } : {}),
    nodes: entry.nodes,
    edges: entry.edges,
  };
  const entries = [next, ...existing].slice(0, MAX_ENTRIES);
  try {
    const payload: StoredPayload = { version: SCHEMA_VERSION, entries };
    window.localStorage.setItem(LS_KEY, JSON.stringify(payload));
  } catch {
    // Quota exceeded or storage disabled — silently ignore. The
    // returned list still reflects what *would* have been stored so
    // in-memory consumers stay consistent within the session.
  }
  return entries;
}

/** Drop the entire history (e.g. user-initiated "Clear recents"). */
export function clearRecentPipelines(): void {
  try {
    window.localStorage.removeItem(LS_KEY);
  } catch {
    // ignore
  }
}
