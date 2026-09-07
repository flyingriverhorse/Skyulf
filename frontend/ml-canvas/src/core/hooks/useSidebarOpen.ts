import { useViewStore } from '../store/useViewStore';
import { useViewport } from './useViewport';

/** Keep layout consumers aligned with responsive defaults and explicit sidebar choices. */
export function useSidebarOpen(): boolean {
  const override = useViewStore((state) => state.sidebarOpenOverride);
  const { width } = useViewport();
  // Tailwind's xl breakpoint leaves room for the library, settings, and graph.
  return override ?? width >= 1280;
}
