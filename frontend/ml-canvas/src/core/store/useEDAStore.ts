/**
 * EDA view + analysis-input state.
 *
 * Consolidates the ~15 local `useState` calls in `EDAPage.tsx` so any
 * sub-component can read or mutate the relevant slice without
 * prop-drilling, and so a dataset switch can clear everything in one
 * call (`resetForDataset`).
 *
 * Out of scope:
 * - Server state (`datasets`, `report`, `history`, `loading`, `error`)
 *   stays on the page — that surface is owned by Phase F #17 (React
 *   Query). Mixing async cache state into zustand here would just be
 *   wasted work the day we migrate.
 */

import { create } from 'zustand';
import type { Filter as ApiFilter } from '../api/eda';

/** Re-exported under a friendlier name; identical to the request payload type. */
export type EDAFilter = ApiFilter;

/** Bivariate / PCA scatter-plot picks, kept together so 3-D / PCA-3-D toggles co-locate with their axes. */
export interface ScatterAxes {
  x: string;
  y: string;
  z: string;
  color: string;
  is3D: boolean;
  isPCA3D: boolean;
}

export interface EDAState {
  // ── Navigation ────────────────────────────────────────────────────
  activeTab: string;

  // ── Analysis inputs ───────────────────────────────────────────────
  selectedDataset: number | null;
  targetCol: string;
  /** "" = auto-detect, "Classification", or "Regression". */
  taskType: string;
  /**
   * Excluded-columns has a draft / applied split: the user toggles
   * checkboxes (`draft`), and only re-runs analysis once they hit
   * "Apply" (`applied`). `excludedDirty` derives the diff.
   */
  excludedColsDraft: string[];
  excludedColsApplied: string[];
  filters: EDAFilter[];

  // ── Visualisation picks ───────────────────────────────────────────
  scatter: ScatterAxes;

  // ── Actions ───────────────────────────────────────────────────────
  setActiveTab: (tab: string) => void;
  setSelectedDataset: (id: number | null) => void;
  setTargetCol: (col: string) => void;
  setTaskType: (type: string) => void;

  /** Toggle a single column in the draft set. Idempotent both ways. */
  toggleExclude: (col: string, exclude: boolean) => void;
  setExcludedDraft: (cols: string[]) => void;
  setExcludedApplied: (cols: string[]) => void;
  /** Copy draft → applied. Caller is responsible for kicking off re-analysis. */
  applyExcluded: () => void;

  addFilter: (filter: EDAFilter) => void;
  removeFilter: (index: number) => void;
  clearFilters: () => void;
  setFilters: (filters: EDAFilter[]) => void;

  setScatter: (patch: Partial<ScatterAxes>) => void;

  /** Wipe per-dataset state when the user switches the selected dataset. */
  resetForDataset: () => void;
}

const EMPTY_SCATTER: ScatterAxes = {
  x: '',
  y: '',
  z: '',
  color: '',
  is3D: false,
  isPCA3D: false,
};

export const useEDAStore = create<EDAState>((set, get) => ({
  activeTab: 'dashboard',
  selectedDataset: null,
  targetCol: '',
  taskType: '',
  excludedColsDraft: [],
  excludedColsApplied: [],
  filters: [],
  scatter: { ...EMPTY_SCATTER },

  setActiveTab: (tab) => set({ activeTab: tab }),
  setSelectedDataset: (id) => set({ selectedDataset: id }),
  setTargetCol: (col) => set({ targetCol: col }),
  setTaskType: (type) => set({ taskType: type }),

  toggleExclude: (col, exclude) =>
    set((state) => {
      const has = state.excludedColsDraft.includes(col);
      if (exclude) {
        return has ? state : { excludedColsDraft: [...state.excludedColsDraft, col] };
      }
      return has
        ? { excludedColsDraft: state.excludedColsDraft.filter((c) => c !== col) }
        : state;
    }),
  setExcludedDraft: (cols) => set({ excludedColsDraft: cols }),
  setExcludedApplied: (cols) => set({ excludedColsApplied: cols }),
  applyExcluded: () => set({ excludedColsApplied: [...get().excludedColsDraft] }),

  addFilter: (filter) => set((state) => ({ filters: [...state.filters, filter] })),
  removeFilter: (index) =>
    set((state) => ({ filters: state.filters.filter((_, i) => i !== index) })),
  clearFilters: () => set({ filters: [] }),
  setFilters: (filters) => set({ filters }),

  setScatter: (patch) => set((state) => ({ scatter: { ...state.scatter, ...patch } })),

  resetForDataset: () =>
    set({
      excludedColsDraft: [],
      excludedColsApplied: [],
      filters: [],
      targetCol: '',
      activeTab: 'dashboard',
      scatter: { ...EMPTY_SCATTER },
    }),
}));

/**
 * Selector — the draft and applied lists differ. EDAPage uses this to
 * decide whether to enable the "Apply Exclusions" button.
 */
export const selectExcludedDirty = (state: EDAState): boolean => {
  if (state.excludedColsApplied.length !== state.excludedColsDraft.length) return true;
  const applied = new Set(state.excludedColsApplied);
  for (const col of state.excludedColsDraft) {
    if (!applied.has(col)) return true;
  }
  return false;
};
