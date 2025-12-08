import { create } from 'zustand';

type ViewType = 'canvas' | 'experiments' | 'inference';

interface ViewState {
  activeView: ViewType;
  setView: (view: ViewType) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  activeView: 'canvas',
  setView: (view) => set({ activeView: view }),
}));
