import { useViewStore } from '../store/useViewStore';
import { TABLET_MAX_WIDTH, useViewport } from './useViewport';

/**
 * Resolve the effective read-only state for the canvas:
 * - Below `lg` (1024 px) we default to read-only (pan/zoom + inspect)
 *   so tablet users can browse pipelines without fighting a layout
 *   designed for >=1024 px.
 * - The user can override either way from the Navbar toggle; that
 *   override wins until they reset it to `auto`.
 */
export function useReadOnlyMode(): boolean {
  const override = useViewStore((s) => s.readOnlyOverride);
  const { isTablet } = useViewport();
  if (override === 'on') return true;
  if (override === 'off') return false;
  return isTablet;
}

/**
 * Non-reactive snapshot of the same logic for use inside imperative
 * keyboard handlers (where calling a hook would break rules-of-hooks).
 * Re-resolves on every keypress so a viewport resize / override change
 * is picked up without re-registering the listener.
 */
export function getReadOnlyMode(): boolean {
  const override = useViewStore.getState().readOnlyOverride;
  if (override === 'on') return true;
  if (override === 'off') return false;
  if (typeof window === 'undefined') return false;
  return window.innerWidth < TABLET_MAX_WIDTH;
}
