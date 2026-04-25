import { useEffect, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useGraphStore } from '../store/useGraphStore';
import { saveCanvasSnapshot } from '../utils/canvasPersistence';

/**
 * Throttled localStorage autosave for the canvas graph.
 *
 * Behaviour:
 * - Subscribes to `nodes` + `edges` from `useGraphStore`.
 * - Coalesces rapid changes (drag, multi-edit) onto a trailing 1 s
 *   timer so we hit `localStorage.setItem` at most ~1× per second.
 * - Skips the initial empty graph so we don't clobber a pending
 *   restore prompt with `{ nodes: [], edges: [] }` before the user
 *   has even decided.
 * - On unmount, flushes any pending write so a quick refresh after the
 *   last edit still persists.
 */
const AUTOSAVE_INTERVAL_MS = 1000;

export function useCanvasAutoSave(): void {
  const { nodes, edges } = useGraphStore(
    useShallow((s) => ({ nodes: s.nodes, edges: s.edges })),
  );

  const timerRef = useRef<number | null>(null);
  const latestRef = useRef({ nodes, edges });
  latestRef.current = { nodes, edges };

  useEffect(() => {
    // Don't autosave an entirely empty canvas — that's the initial
    // mount before any nodes are added and would overwrite a stored
    // snapshot the user might still want to restore.
    if (nodes.length === 0 && edges.length === 0) return;

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      saveCanvasSnapshot(latestRef.current.nodes, latestRef.current.edges);
      timerRef.current = null;
    }, AUTOSAVE_INTERVAL_MS);

    return () => {
      // Effect re-runs on every store change; the new effect resets
      // the timer above. We only flush on true unmount (handled in
      // the outer effect below).
    };
  }, [nodes, edges]);

  // Flush-on-unmount guard: if a write is pending when the canvas
  // unmounts (e.g. user navigates to Experiments mid-edit), commit it
  // synchronously so nothing is lost.
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
        const { nodes: n, edges: e } = latestRef.current;
        if (n.length > 0 || e.length > 0) {
          saveCanvasSnapshot(n, e);
        }
      }
    };
  }, []);
}
