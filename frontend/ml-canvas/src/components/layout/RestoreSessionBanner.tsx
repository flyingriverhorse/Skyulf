import React, { useEffect, useState } from 'react';
import { History, X } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import {
  clearCanvasSnapshot,
  loadCanvasSnapshot,
  type CanvasSnapshot,
} from '../../core/utils/canvasPersistence';
import { clickableProps } from '../../core/utils/a11y';

/**
 * One-shot prompt that surfaces a previously auto-saved canvas when
 * the user reopens the app with an empty graph. Restores `nodes` /
 * `edges` from `localStorage` on confirmation, or wipes the snapshot
 * on dismiss. Stays hidden the rest of the session.
 *
 * Pairs with `useCanvasAutoSave`.
 */
export const RestoreSessionBanner: React.FC = () => {
  const setGraph = useGraphStore((s) => s.setGraph);
  const hasNodes = useGraphStore((s) => s.nodes.length > 0);

  const [snapshot, setSnapshot] = useState<CanvasSnapshot | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Only probe localStorage once on mount, and only show the banner
    // when the canvas is currently empty (otherwise the user already
    // started fresh — don't second-guess them).
    if (hasNodes) return;
    const snap = loadCanvasSnapshot();
    if (snap && (snap.nodes.length > 0 || snap.edges.length > 0)) {
      setSnapshot(snap);
    }
    // Intentionally only run once: subsequent edits shouldn't
    // re-trigger the prompt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (dismissed || !snapshot || hasNodes) return null;

  const handleRestore = (): void => {
    setGraph(snapshot.nodes, snapshot.edges);
    setDismissed(true);
  };

  const handleDiscard = (): void => {
    clearCanvasSnapshot();
    setSnapshot(null);
    setDismissed(true);
  };

  // Format "5 minutes ago" without pulling in date-fns; coarse buckets
  // are plenty for an autosave hint.
  const minutesAgo = Math.max(
    0,
    Math.round((Date.now() - new Date(snapshot.savedAt).getTime()) / 60000),
  );
  const relative =
    minutesAgo < 1
      ? 'just now'
      : minutesAgo < 60
        ? `${minutesAgo} min ago`
        : minutesAgo < 60 * 24
          ? `${Math.round(minutesAgo / 60)} h ago`
          : `${Math.round(minutesAgo / (60 * 24))} d ago`;

  return (
    <div
      role="status"
      aria-live="polite"
      // Anchored to the bottom of the canvas viewport so it doesn't
      // collide with the toolbar / Run buttons up top. Sits just above
      // the Results panel collapsed bar (h-10) when present.
      className="absolute bottom-14 left-1/2 -translate-x-1/2 z-30 flex items-center gap-3 px-4 py-2 rounded-md border bg-background/95 backdrop-blur shadow-lg text-sm animate-in fade-in slide-in-from-bottom-2"
    >
      <History className="w-4 h-4 text-primary" aria-hidden="true" />
      <span>
        Restore previous session?{' '}
        <span className="text-muted-foreground">
          {snapshot.nodes.length} node{snapshot.nodes.length === 1 ? '' : 's'} · saved {relative}
        </span>
      </span>
      <button
        onClick={handleRestore}
        className="px-2.5 py-1 rounded bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 focus-ring"
      >
        Restore
      </button>
      <button
        onClick={handleDiscard}
        className="px-2.5 py-1 rounded border text-xs font-medium hover:bg-accent focus-ring"
      >
        Discard
      </button>
      <span
        {...clickableProps(() => setDismissed(true))}
        className="ml-1 p-1 rounded hover:bg-accent text-muted-foreground cursor-pointer focus-ring"
        aria-label="Dismiss restore prompt"
      >
        <X className="w-3.5 h-3.5" />
      </span>
    </div>
  );
};
