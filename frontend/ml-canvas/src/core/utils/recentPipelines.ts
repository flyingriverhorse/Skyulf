/**
 * Ring buffer of recently saved pipelines, persisted in `localStorage`.
 *
 * Complements `canvasPersistence.ts` (single-slot autosave) by keeping a
 * named history the user can scroll back through from the Toolbar.
 *
 * Schema is intentionally tiny so quota is rarely a concern; unpinned
 * entries are dropped FIFO once the buffer is full. Pinned entries are
 * exempt from the cap so the user can keep a known-good snapshot
 * indefinitely. All writes are best-effort — a quota or serialization
 * error must never break Save.
 */
import type { Edge, Node } from '@xyflow/react';

const LS_KEY = 'skyulf:canvas:recent:v1';
const SCHEMA_VERSION = 1;
const MAX_UNPINNED = 5;

export interface RecentPipelineEntry {
  /** Stable id used as React key and to address rename/pin/delete ops. */
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
  /** Pinned entries are exempt from the FIFO cap. */
  pinned?: boolean;
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

function writeEntries(entries: RecentPipelineEntry[]): void {
  try {
    const payload: StoredPayload = { version: SCHEMA_VERSION, entries };
    window.localStorage.setItem(LS_KEY, JSON.stringify(payload));
  } catch {
    // Quota exceeded or storage disabled — silently ignore.
  }
}

/** Apply the FIFO cap to unpinned entries while preserving every pinned
 *  entry. Order is left untouched (caller decides recency ordering). */
function capUnpinned(entries: RecentPipelineEntry[]): RecentPipelineEntry[] {
  let kept = 0;
  return entries.filter((e) => {
    if (e.pinned) return true;
    kept += 1;
    return kept <= MAX_UNPINNED;
  });
}

/** Read the current list (pinned first, then unpinned by recency).
 *  Returns `[]` when nothing is stored or the payload is corrupt. */
export function getRecentPipelines(): RecentPipelineEntry[] {
  const entries = readRaw()?.entries ?? [];
  // Stable sort: pinned entries float to the top, recency preserved
  // within each group.
  return [...entries].sort(
    (a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)),
  );
}

/** Push a new entry. Same-name entries are deduped (newest wins, pin
 *  state inherited from the prior entry) so the buffer doesn't fill up
 *  with re-saves of the same pipeline. Returns the resulting list. */
export function pushRecentPipeline(
  entry: Omit<RecentPipelineEntry, 'id' | 'savedAt'> & { savedAt?: string },
): RecentPipelineEntry[] {
  const all = readRaw()?.entries ?? [];
  const sameName = all.find(
    (e) => e.name.trim().toLowerCase() === entry.name.trim().toLowerCase(),
  );
  const remaining = all.filter((e) => e !== sameName);
  const inheritedPin = entry.pinned ?? sameName?.pinned ?? false;
  const next: RecentPipelineEntry = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    name: entry.name,
    savedAt: entry.savedAt ?? new Date().toISOString(),
    ...(entry.datasetId !== undefined ? { datasetId: entry.datasetId } : {}),
    ...(entry.datasetName !== undefined ? { datasetName: entry.datasetName } : {}),
    ...(inheritedPin ? { pinned: true as const } : {}),
    nodes: entry.nodes,
    edges: entry.edges,
  };
  const entries = capUnpinned([next, ...remaining]);
  writeEntries(entries);
  return entries;
}

/** Toggle the pin state for an entry. Returns the updated list, or the
 *  current list unchanged if the id is unknown. */
export function togglePinRecentPipeline(id: string): RecentPipelineEntry[] {
  const all = readRaw()?.entries ?? [];
  if (!all.some((e) => e.id === id)) return all;
  const updated = all.map((e) => {
    if (e.id !== id) return e;
    if (e.pinned) {
      // Strip the field entirely so the JSON stays minimal and the
      // exactOptionalPropertyTypes invariant ("absent vs undefined") holds.
      const { pinned: _drop, ...rest } = e;
      return rest as RecentPipelineEntry;
    }
    return { ...e, pinned: true as const };
  });
  writeEntries(updated);
  return updated;
}

/** Rename an entry. No-op (returns current list) when the new name is
 *  empty or already used by another entry. */
export function renameRecentPipeline(id: string, newName: string): RecentPipelineEntry[] {
  const all = readRaw()?.entries ?? [];
  const trimmed = newName.trim();
  if (!trimmed) return all;
  const clash = all.some(
    (e) => e.id !== id && e.name.trim().toLowerCase() === trimmed.toLowerCase(),
  );
  if (clash) return all;
  const updated = all.map((e) => (e.id === id ? { ...e, name: trimmed } : e));
  writeEntries(updated);
  return updated;
}

/** Delete a single entry by id. Pinned entries are not protected from
 *  explicit user deletion. */
export function deleteRecentPipeline(id: string): RecentPipelineEntry[] {
  const all = readRaw()?.entries ?? [];
  const updated = all.filter((e) => e.id !== id);
  writeEntries(updated);
  return updated;
}

/** Drop the entire history (e.g. user-initiated "Clear recents").
 *  Pinned entries are removed — explicit user action, not eviction. */
export function clearRecentPipelines(): void {
  try {
    window.localStorage.removeItem(LS_KEY);
  } catch {
    // ignore
  }
}
