/**
 * Notifications store — retains app messages, execution feedback, and pipeline warnings
 * for the navbar bell so users can review them without transient popups.
 *
 * Populated by canvas run controls and `useExecutionWarnings` whenever the
 * canvas receives an `executionResult.node_warnings` payload. Surfaced by
 * `NotificationCenter` (bell icon in the navbar).
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { NodeWarning } from '../api/client';
import type { SubmittedRun } from '../types/runFeedback';

export type ExecutionNotificationAction = { type: 'preview' } | { type: 'jobs'; run: SubmittedRun };

export interface StoredNotification extends NodeWarning {
  /** Unique id so React lists are stable across re-renders. */
  id: string;
  /** ms-since-epoch timestamp of when the notification was buffered. */
  ts: number;
  /** False until the user opens the notification panel. */
  read: boolean;
  /** Serializable destination for execution feedback; absent on legacy warnings. */
  action?: ExecutionNotificationAction;
}

interface NotificationsState {
  items: StoredNotification[];
  /** Add a batch of warnings (deduped against existing items by message+node). */
  addMany: (warnings: NodeWarning[]) => void;
  /** Refresh a repeated app message as unread without accumulating duplicate rows. */
  addAppMessage: (level: string, message: string) => void;
  /** Insert or replace one execution notice while keeping unchanged messages read. */
  upsertExecution: (id: string, message: string, action: ExecutionNotificationAction, level?: string) => void;
  /** Mark every item as read (called when the panel opens). */
  markAllRead: () => void;
  /** Drop a single item by id. */
  dismiss: (id: string) => void;
  /** Drop everything. */
  clear: () => void;
}

const MAX_ITEMS = 100;

/** Generate a notification ID from 128 bits of cryptographic randomness. */
function createNotificationId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}

export const useNotificationsStore = create<NotificationsState>()(
  persist(
    (set) => ({
  items: [],
  addAppMessage: (level, message) => set(state => {
    const previous = state.items.find(item => item.logger === 'app' && item.level === level && item.message === message);
    const id = previous?.id ?? createNotificationId();
    const item: StoredNotification = {
      id, level, message, node_id: null, node_type: null, logger: 'app', ts: Date.now(), read: false,
    };
    return { items: [item, ...state.items.filter(existing => existing.id !== id)].slice(0, MAX_ITEMS) };
  }),
  upsertExecution: (id, message, action, level = 'info') =>
    set(state => {
      const previous = state.items.find(item => item.id === id);
      const changed = !previous || previous.message !== message || previous.level !== level;
      const item: StoredNotification = {
        id, message, action, level, node_id: null, node_type: null, logger: 'canvas',
        ts: changed ? Date.now() : previous.ts,
        read: changed ? false : previous.read,
      };
      return { items: changed
        ? [item, ...state.items.filter(existing => existing.id !== id)].slice(0, MAX_ITEMS)
        : state.items.map(existing => existing.id === id ? item : existing),
      };
    }),
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
          id: createNotificationId(),
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
