import { create } from 'zustand';

type ViewType = 'canvas' | 'experiments' | 'inference';

// `auto` defers to viewport width (read-only below the lg breakpoint).
// `on`/`off` are explicit user overrides from the Navbar toggle.
export type ReadOnlyOverride = 'auto' | 'on' | 'off';

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
}));
