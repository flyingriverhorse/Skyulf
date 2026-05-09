/**
 * Notifications store — buffers per-pipeline-run warnings (toasts come and
 * go in seconds; users want a place to scroll back through what happened).
 *
 * Populated by `useExecutionWarnings` whenever the canvas receives a new
 * `executionResult.node_warnings` payload from the backend. Surfaced by
 * `NotificationCenter` (bell icon in the navbar).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { NodeWarning } from '../api/client';

export interface StoredNotification extends NodeWarning {
  /** Unique id so React lists are stable across re-renders. */
  id: string;
  /** ms-since-epoch timestamp of when the notification was buffered. */
  ts: number;
  /** False until the user opens the notification panel. */
  read: boolean;
}

interface NotificationsState {
  items: StoredNotification[];
  /** Add a batch of warnings (deduped against existing items by message+node). */
  addMany: (warnings: NodeWarning[]) => void;
  /** Mark every item as read (called when the panel opens). */
  markAllRead: () => void;
  /** Drop a single item by id. */
  dismiss: (id: string) => void;
  /** Drop everything. */
  clear: () => void;
}

const MAX_ITEMS = 100;

export const useNotificationsStore = create<NotificationsState>()(
  persist(
    (set) => ({
  items: [],
  addMany: (warnings) =>
    set((state) => {
      if (warnings.length === 0) return state;
      // Dedup against the existing buffer on (node_id, message) so re-running
      // the same pipeline doesn't pile up identical entries.
      const existingKeys = new Set(
        state.items.map((it) => `${it.node_id ?? ''}::${it.message}`),
      );
      const fresh: StoredNotification[] = [];
      for (const w of warnings) {
        const key = `${w.node_id ?? ''}::${w.message}`;
        if (existingKeys.has(key)) continue;
        existingKeys.add(key);
        fresh.push({
          ...w,
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          ts: Date.now(),
          read: false,
        });
      }
      if (fresh.length === 0) return state;
      // Newest first; cap to MAX_ITEMS so the buffer can't grow unbounded
      // across long sessions.
      const next = [...fresh, ...state.items].slice(0, MAX_ITEMS);
      return { items: next };
    }),
  markAllRead: () =>
    set((state) => ({ items: state.items.map((it) => ({ ...it, read: true })) })),
  dismiss: (id) =>
    set((state) => ({ items: state.items.filter((it) => it.id !== id) })),
  clear: () => set({ items: [] }),
    }),
    {
      name: 'skyulf-notifications',
      // Only persist the items array; functions are recreated from the store factory.
      partialize: (state) => ({ items: state.items }),
    },
  ),
);

/** Convenience selector for the navbar badge. */
export const selectUnreadCount = (state: NotificationsState): number =>
  state.items.filter((it) => !it.read).length;
