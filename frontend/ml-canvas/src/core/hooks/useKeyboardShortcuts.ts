import { useEffect } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { useViewStore } from '../store/useViewStore';

/**
 * Custom event name fired when the user presses Ctrl/Cmd+Enter on
 * the canvas. The Toolbar listens for this and triggers its existing
 * "Run Preview" handler — keeping run logic where it already lives
 * avoids duplicating pipeline-conversion code into the hook.
 */
export const RUN_PREVIEW_EVENT = 'skyulf:run-preview';

/**
 * Custom event fired when any UI affordance (e.g. the Toolbar's
 * Keyboard button) wants to open the shortcuts cheatsheet. The
 * overlay state lives in `MainLayout`; this lets distant components
 * trigger it without prop-drilling or a dedicated store.
 */
export const SHOW_SHORTCUTS_EVENT = 'skyulf:show-shortcuts';

/**
 * Custom event fired when the user presses Ctrl/Cmd+K (or clicks the
 * palette button). Picked up by the global `<CommandPalette />` mounted
 * in `MainLayout`.
 */
export const SHOW_PALETTE_EVENT = 'skyulf:show-palette';

/**
 * Custom event the palette dispatches when the user picks a node.
 * `FlowCanvas` listens and inserts the node at the current viewport
 * center using `useReactFlow().screenToFlowPosition` (only available
 * inside `<ReactFlowProvider>`).
 */
export const ADD_NODE_AT_CENTER_EVENT = 'skyulf:add-node-at-center';

export interface AddNodeAtCenterDetail {
  type: string;
}

interface ShortcutOptions {
  /** Toggles the `?` shortcuts cheatsheet overlay. */
  onToggleHelp: () => void;
  /** Closes the cheatsheet (and any other Esc-dismissible UI). */
  onCloseHelp: () => void;
}

const isEditableTarget = (target: EventTarget | null): boolean => {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    el.isContentEditable === true
  );
};

/**
 * Global canvas keyboard shortcuts. Registered once at the layout
 * level so they work regardless of which panel has focus, but skipped
 * when the user is typing in a form field so we don't fight native
 * input behaviour.
 *
 * Shortcut map:
 *  - `Ctrl/Cmd+D`  → duplicate selected nodes
 *  - `Ctrl/Cmd+Enter` → run preview (dispatched as a CustomEvent)
 *  - `?` (Shift+/) → open shortcuts overlay
 *  - `Esc`         → close overlay / collapse expanded properties panel
 *
 * Delete/Backspace is left to React Flow's built-in `deleteKeyCode`,
 * and Ctrl+Z / Ctrl+Shift+Z is handled in the Toolbar against the
 * zundo temporal store.
 */
export function useKeyboardShortcuts({
  onToggleHelp,
  onCloseHelp,
}: ShortcutOptions): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      if (isEditableTarget(e.target)) return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();

      // Esc: close overlay first; then collapse expanded properties panel.
      if (e.key === 'Escape') {
        const view = useViewStore.getState();
        if (view.isPropertiesPanelExpanded) {
          view.setPropertiesPanelExpanded(false);
        }
        onCloseHelp();
        return;
      }

      // ? (Shift+/) → toggle help overlay. Match on `?` directly so
      // we cover both US and non-US layouts (where Shift+/ may map to
      // a different key).
      if (e.key === '?') {
        e.preventDefault();
        onToggleHelp();
        return;
      }

      if (mod && key === 'd') {
        e.preventDefault();
        useGraphStore.getState().duplicateSelectedNodes();
        return;
      }

      if (mod && (e.key === 'Enter' || key === 'enter')) {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT));
        return;
      }

      // Ctrl/Cmd+K → open the command palette. Modifier-gated so plain
      // `k` typed elsewhere stays a no-op.
      if (mod && key === 'k') {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT));
        return;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onToggleHelp, onCloseHelp]);
}
