/**
 * Bell icon + dropdown panel for the navbar. Shows a count badge of unread
 * per-node pipeline warnings. Clicking a notification opens a detail modal.
 * Backed by `useNotificationsStore`.
 */

import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router-dom';
import { Bell, X, AlertTriangle, Info, AlertCircle, ExternalLink } from 'lucide-react';
import {
  useNotificationsStore,
  type StoredNotification,
} from '../../core/store/useNotificationsStore';
import { toast } from '../../core/toast';

const formatTime = (ts: number): string => {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const formatDateTime = (ts: number): string => {
  const d = new Date(ts);
  return d.toLocaleString([], {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
};

const LevelIcon: React.FC<{ level: string }> = ({ level }) => {
  const l = level.toLowerCase();
  if (l === 'error') return <AlertCircle className="w-4 h-4 text-red-500" />;
  if (l === 'warning' || l === 'warn') return <AlertTriangle className="w-4 h-4 text-amber-500" />;
  return <Info className="w-4 h-4 text-blue-500" />;
};

// ---------------------------------------------------------------------------
// Detail modal
// ---------------------------------------------------------------------------
const NotificationDetailModal: React.FC<{
  item: StoredNotification;
  onClose: () => void;
  onViewErrorLog: () => void;
}> = ({ item, onClose, onViewErrorLog }) => {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const levelColour =
    item.level.toLowerCase() === 'error'
      ? 'text-red-600 dark:text-red-400 bg-red-500/10'
      : item.level.toLowerCase().startsWith('warn')
      ? 'text-amber-700 dark:text-amber-400 bg-amber-500/10'
      : 'text-blue-600 dark:text-blue-400 bg-blue-500/10';

  return (
    // Backdrop — dismiss on click outside
    <div
      role="presentation"
      className="fixed inset-0 z-[99999] flex items-center justify-center bg-black/40 backdrop-blur-[2px]"
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="relative w-full max-w-lg mx-4 bg-card border rounded-xl shadow-2xl flex flex-col max-h-[80vh]">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b">
          <LevelIcon level={item.level} />
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-semibold truncate">
              {item.node_type ?? 'Pipeline notification'}
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">{formatDateTime(item.ts)}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close detail"
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto px-5 py-4 flex flex-col gap-4">
          {/* Level badge */}
          <div className="flex items-center gap-2">
            <span className={`text-xs font-semibold px-2 py-0.5 rounded ${levelColour}`}>
              {item.level.toUpperCase()}
            </span>
            {item.logger && (
              <span className="text-xs text-muted-foreground font-mono">{item.logger}</span>
            )}
          </div>

          {/* Message */}
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Message</p>
            <div className="bg-muted/50 rounded-md px-3 py-2 text-sm leading-relaxed break-words whitespace-pre-wrap font-mono">
              {item.message}
            </div>
          </div>

          {/* Node info */}
          {(item.node_id || item.node_type) && (
            <div className="grid grid-cols-2 gap-3">
              {item.node_type && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Node type</p>
                  <p className="text-sm font-mono bg-muted/50 rounded px-2 py-1">{item.node_type}</p>
                </div>
              )}
              {item.node_id && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Node ID</p>
                  <p className="text-xs font-mono bg-muted/50 rounded px-2 py-1 truncate" title={item.node_id}>
                    {item.node_id}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 text-sm bg-secondary hover:bg-secondary/80 rounded-md transition-colors"
          >
            Close
          </button>
          <button
            type="button"
            onClick={onViewErrorLog}
            className="flex items-center gap-1.5 px-4 py-1.5 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 rounded-md transition-colors border border-red-200 dark:border-red-800"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            View in Error Log
          </button>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export const NotificationCenter: React.FC = () => {
  const items = useNotificationsStore((s) => s.items);
  const markAllRead = useNotificationsStore((s) => s.markAllRead);
  const dismiss = useNotificationsStore((s) => s.dismiss);
  const clear = useNotificationsStore((s) => s.clear);

  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<StoredNotification | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  // Auto-open removed — badge count is sufficient signal; panel opens on click only.
  useEffect(() => {
    if (!open) return;
    const onDoc = (ev: MouseEvent): void => {
      if (ref.current && !ref.current.contains(ev.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (ev: KeyboardEvent): void => {
      if (ev.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const togglePanel = (): void => {
    setOpen((prev) => {
      const next = !prev;
      if (next) markAllRead();
      return next;
    });
  };

  return (
    <>
      <div ref={ref} className="relative">
        <button
          type="button"
          onClick={togglePanel}
          title="Pipeline notifications"
          aria-label={`Notifications${items.length > 0 ? ` (${items.length})` : ''}`}
          className="relative flex items-center justify-center w-9 h-9 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition-colors"
        >
          <Bell className="w-4 h-4" />
          {items.length > 0 && (
            <span
              className="absolute -top-0.5 -right-0.5 min-w-[16px] h-[16px] px-1 rounded-full bg-amber-500 text-white text-[10px] font-semibold flex items-center justify-center"
              aria-hidden="true"
            >
              {items.length > 99 ? '99+' : items.length}
            </span>
          )}
        </button>

        {open && (
          <div className="absolute right-0 top-full mt-2 w-96 max-h-[28rem] bg-card border rounded-lg shadow-xl z-[9999] flex flex-col">
            <div className="flex items-center justify-between px-3 py-2 border-b">
              <span className="text-sm font-medium">Notifications</span>
              <div className="flex items-center gap-2">
                {items.length > 0 && (
                  <button
                    type="button"
                    onClick={() => { clear(); toast.dismissAll(); }}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Clear all
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  aria-label="Close"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {items.length === 0 ? (
              <div className="flex-1 flex items-center justify-center px-4 py-12 text-sm text-muted-foreground">
                No notifications. Pipeline advisories will show up here.
              </div>
            ) : (
              <ul className="overflow-y-auto divide-y">
                {items.map((it) => (
                  <li key={it.id} className="group">
                    {/* Click the row to open the detail modal */}
                    <button
                      type="button"
                      onClick={() => { setSelected(it); setOpen(false); }}
                      className="w-full text-left px-3 py-2 hover:bg-accent/40 flex items-start justify-between gap-2"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 text-xs">
                          <LevelIcon level={it.level} />
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-400 font-medium">
                            {it.level}
                          </span>
                          {it.node_type && (
                            <span className="text-muted-foreground truncate">{it.node_type}</span>
                          )}
                          <span className="ml-auto text-muted-foreground shrink-0">
                            {formatTime(it.ts)}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-foreground line-clamp-2 break-words">
                          {it.message}
                        </p>
                        {it.node_id && (
                          <p className="mt-0.5 text-[10px] text-muted-foreground font-mono truncate">
                            {it.node_id}
                          </p>
                        )}
                        <p className="mt-1 text-[10px] text-muted-foreground/60 italic">
                          Click to see full details
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); dismiss(it.id); }}
                        aria-label="Dismiss"
                        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground shrink-0 mt-0.5"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Detail modal rendered via portal so it escapes parent CSS transforms */}
      {selected && createPortal(
        <NotificationDetailModal
          item={selected}
          onClose={() => setSelected(null)}
          onViewErrorLog={() => { setSelected(null); setOpen(false); navigate('/errors'); }}
        />,
        document.body,
      )}
    </>
  );
};
