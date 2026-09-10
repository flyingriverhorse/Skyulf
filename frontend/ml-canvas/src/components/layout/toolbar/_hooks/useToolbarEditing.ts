import { useCallback, useEffect } from 'react';
import { useGraphStore, useTemporalStore } from '../../../../core/store/useGraphStore';
import { getReadOnlyMode } from '../../../../core/hooks/useReadOnlyMode';
import { useConfirm } from '../../../shared';

/** Respect native editing, modifier keys, and the current read-only setting. */
function ignoreHistoryShortcut(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null;
  const editable = ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName ?? '')
    || target?.isContentEditable === true;
  return editable || !(event.ctrlKey || event.metaKey) || getReadOnlyMode();
}

/** Preserve recoverable graph editing and native-input history shortcuts. */
export function useToolbarEditing(readOnly: boolean) {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setGraph = useGraphStore((state) => state.setGraph);

  // Undo/redo from the temporal substore (zundo). Separate selectors so
  // the toolbar only re-renders when the counts flip across zero.
  const undo = useTemporalStore((s) => s.undo);
  const redo = useTemporalStore((s) => s.redo);
  const canUndo = useTemporalStore((s) => s.pastStates.length > 0);
  const canRedo = useTemporalStore((s) => s.futureStates.length > 0);

  // Clear Canvas: wipe every node + edge after explicit confirmation.
  // Lives next to Undo/Redo because it's the canonical "reset" action;
  // Ctrl+Z still restores the previous state via zundo so this is recoverable.
  const confirm = useConfirm();
  const canClear = !readOnly && (nodes.length > 0 || edges.length > 0);
  const handleClearCanvas = useCallback(async (): Promise<void> => {
    if (nodes.length === 0 && edges.length === 0) return;
    const ok = await confirm({
      title: 'Clear the canvas?',
      message: `Remove all ${nodes.length} node(s) and ${edges.length} edge(s)? You can undo with Ctrl+Z.`,
      confirmLabel: 'Clear canvas',
      variant: 'danger',
    });
    if (ok) setGraph([], []);
  }, [nodes.length, edges.length, confirm, setGraph]);

  // Global undo/redo hotkeys. Skip when focus is in a text input so we
  // don't fight native input undo.
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      if (ignoreHistoryShortcut(e)) return;
      const key = e.key.toLowerCase();
      if (key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((key === 'z' && e.shiftKey) || key === 'y') {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo]);

  return { nodes, edges, setGraph, undo, redo, canUndo, canRedo, canClear, handleClearCanvas };
}
