import { create } from 'zustand';

type ViewType = 'canvas' | 'experiments' | 'inference';

interface ViewState {
  activeView: ViewType;
  setView: (view: ViewType) => void;
  isSidebarOpen: boolean;
  setSidebarOpen: (isOpen: boolean) => void;
  isPropertiesPanelExpanded: boolean;
  setPropertiesPanelExpanded: (isExpanded: boolean) => void;
  isResultsPanelExpanded: boolean;
  setResultsPanelExpanded: (isExpanded: boolean) => void;
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
}));
