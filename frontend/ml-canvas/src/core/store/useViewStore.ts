import { create } from 'zustand';

type ViewType = 'canvas' | 'experiments' | 'inference';

// `auto` defers to viewport width (read-only below the lg breakpoint).
// `on`/`off` are explicit user overrides from the Navbar toggle.
export type ReadOnlyOverride = 'auto' | 'on' | 'off';

const PERF_OVERLAY_STORAGE_KEY = 'skyulf:perf-overlay-enabled';

const readPerfOverlayPreference = (): boolean => {
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage.getItem(PERF_OVERLAY_STORAGE_KEY) === '1';
  } catch {
    // Storage blocked (private mode, quota, etc.) — default off.
    return false;
  }
};

const writePerfOverlayPreference = (enabled: boolean): void => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(PERF_OVERLAY_STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    // Best-effort; not surfacing an error for a UI preference.
  }
};

interface ViewState {
  activeView: ViewType;
  setView: (view: ViewType) => void;
  isSidebarOpen: boolean;
  setSidebarOpen: (isOpen: boolean) => void;
  isPropertiesPanelExpanded: boolean;
  setPropertiesPanelExpanded: (isExpanded: boolean) => void;
  isResultsPanelExpanded: boolean;
  setResultsPanelExpanded: (isExpanded: boolean) => void;
  readOnlyOverride: ReadOnlyOverride;
  setReadOnlyOverride: (mode: ReadOnlyOverride) => void;
  /** L4 perf overlay: when true, every node card whose last run has a
   * known wall-clock duration shows a colored ring (green < 100 ms,
   * amber < 1 s, red ≥ 1 s) plus a tooltip with the exact ms. Off
   * by default — opt-in via the Toolbar gauge button. */
  perfOverlayEnabled: boolean;
  setPerfOverlayEnabled: (enabled: boolean) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  activeView: 'canvas',
  setView: (view) => set({ activeView: view }),
  isSidebarOpen: true,
  setSidebarOpen: (isOpen) => set({ isSidebarOpen: isOpen }),
  isPropertiesPanelExpanded: false,
  setPropertiesPanelExpanded: (isExpanded) => set({ isPropertiesPanelExpanded: isExpanded }),
  isResultsPanelExpanded: true,
  setResultsPanelExpanded: (isExpanded) => set({ isResultsPanelExpanded: isExpanded }),
  readOnlyOverride: 'auto',
  setReadOnlyOverride: (mode) => set({ readOnlyOverride: mode }),
  perfOverlayEnabled: readPerfOverlayPreference(),
  setPerfOverlayEnabled: (enabled) => {
    writePerfOverlayPreference(enabled);
    set({ perfOverlayEnabled: enabled });
  },
}));
