import { useEffect } from 'react';

/**
 * Close a popover/dropdown when Esc is pressed or a mousedown lands outside `ref`.
 * Uses mousedown (not click) so the menu closes before a click on another trigger
 * can re-open it — avoids the "stuck open" feel when toggling between dropdowns.
 * Accepts a single ref or several refs (e.g. a trigger cluster plus a popover that
 * renders outside the cluster's stacking context).
 */
export function useDismissable(
  open: boolean,
  close: () => void,
  ref: React.RefObject<HTMLElement | null> | React.RefObject<HTMLElement | null>[],
): void {
  useEffect(() => {
    if (!open) return;
    const refs = Array.isArray(ref) ? ref : [ref];
    const onMouseDown = (e: MouseEvent): void => {
      const target = e.target as unknown as globalThis.Node;
      // `Node` is shadowed by React Flow's node type in Toolbar.tsx;
      // route through globalThis to reach the DOM Node constructor.
      if (refs.some((r) => r.current?.contains(target))) return;
      close();
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
