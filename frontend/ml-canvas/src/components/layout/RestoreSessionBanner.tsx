import React, { useEffect, useState } from 'react';
import { History, X, AlertTriangle } from 'lucide-react';
import { useGraphStore } from '../../core/store/useGraphStore';
import {
  clearCanvasSnapshot,
  loadCanvasSnapshotDiagnostic,
  type CanvasSnapshot,
} from '../../core/utils/canvasPersistence';
import {
  describeAutosaveUnavailable,
  RECOVERY_KIND_LABEL,
  type AutosaveUnavailable,
} from '../../core/utils/canvasRecovery';
import { FIT_VIEW_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { clickableProps } from '../../core/utils/a11y';

/**
 * Recovery entry point that surfaces the autosaved canvas — or explains why
 * it can't be restored — whenever the user reopens the app with an empty
 * graph. Restores `nodes`/`edges` from `localStorage` on confirmation, or
 * wipes the snapshot on dismiss. Re-probes any time the canvas becomes
 * empty (e.g. after "Clear canvas"), not just on first mount, so clearing
 * the graph doesn't permanently suppress the prompt for the session.
 *
 * Pairs with `useCanvasAutoSave`.
 */
export const RestoreSessionBanner: React.FC = () => {
  const setGraph = useGraphStore((s) => s.setGraph);
  const hasNodes = useGraphStore((s) => s.nodes.length > 0);

  const [snapshot, setSnapshot] = useState<CanvasSnapshot | null>(null);
  const [unavailable, setUnavailable] = useState<AutosaveUnavailable | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    // Only show the banner while the canvas is currently empty (otherwise
    // the user already started fresh — never second-guess a nonempty
    // graph). Re-runs whenever the graph transitions to empty so a
    // mid-session "Clear canvas" gets a fresh chance to offer recovery.
    if (hasNodes) {
      setDismissed(false);
      return;
    }
    const diagnostic = loadCanvasSnapshotDiagnostic();
    if (diagnostic.status === 'available' && diagnostic.snapshot.nodes.length + diagnostic.snapshot.edges.length > 0) {
      setSnapshot(diagnostic.snapshot);
      setUnavailable(null);
    } else {
      setSnapshot(null);
      setUnavailable(describeAutosaveUnavailable(diagnostic));
    }
  }, [hasNodes]);

  if (dismissed || hasNodes || (!snapshot && !unavailable)) return null;

  const handleRestore = (): void => {
    if (!snapshot) return;
    setGraph(snapshot.nodes, snapshot.edges);
    setDismissed(true);
    // CAN-003: restoring a source must focus the result, not leave the
    // user staring at wherever the viewport happened to be.
    window.dispatchEvent(new CustomEvent(FIT_VIEW_EVENT));
  };

  const handleDiscard = (): void => {
    clearCanvasSnapshot();
    setSnapshot(null);
    setDismissed(true);
  };

  if (unavailable) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="absolute bottom-14 left-1/2 -translate-x-1/2 z-30 flex items-center gap-3 px-4 py-2 rounded-md border bg-background/95 backdrop-blur shadow-lg text-sm animate-in fade-in slide-in-from-bottom-2 max-w-lg"
      >
        <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0" aria-hidden="true" />
        <span className="text-muted-foreground">{unavailable.message}</span>
        <span
          {...clickableProps(() => setDismissed(true))}
          className="ml-1 p-1 rounded hover:bg-accent text-muted-foreground cursor-pointer focus-ring flex-shrink-0"
          aria-label="Dismiss autosave notice"
        >
          <X className="w-3.5 h-3.5" />
        </span>
      </div>
    );
  }

  if (!snapshot) return null;

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
        <span className="px-1.5 py-0.5 mr-1.5 rounded bg-primary/10 text-primary text-xs font-medium">
          {RECOVERY_KIND_LABEL.autosave}
        </span>
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
