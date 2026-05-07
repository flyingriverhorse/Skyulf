import { useEffect } from 'react';

/**
 * Close a popover/dropdown when Esc is pressed or a mousedown lands outside `ref`.
 * Uses mousedown (not click) so the menu closes before a click on another trigger
 * can re-open it — avoids the "stuck open" feel when toggling between dropdowns.
 */
export function useDismissable(
  open: boolean,
  close: () => void,
  ref: React.RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent): void => {
      const node = ref.current;
      // `Node` is shadowed by React Flow's node type in Toolbar.tsx;
      // route through globalThis to reach the DOM Node constructor.
      if (node && !node.contains(e.target as unknown as globalThis.Node)) close();
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') close();
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, close, ref]);
}
