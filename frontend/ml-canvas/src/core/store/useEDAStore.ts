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
  filtersDraft: EDAFilter[];
  filtersApplied: EDAFilter[];

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

  addFilterDraft: (filter: EDAFilter) => void;
  removeFilterDraft: (index: number) => void;
  clearFiltersDraft: () => void;
  setFiltersDraft: (filters: EDAFilter[]) => void;
  setFiltersApplied: (filters: EDAFilter[]) => void;
  applyFilters: () => void;

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

const filterSignature = (filter: EDAFilter): string =>
  `${filter.column}\u001f${filter.operator}\u001f${JSON.stringify(filter.value)}`;

const areFiltersEqual = (left: EDAFilter[], right: EDAFilter[]): boolean => {
  if (left.length !== right.length) return false;
  return left.every((filter, index) => filterSignature(filter) === filterSignature(right[index]!));
};

export const useEDAStore = create<EDAState>((set, get) => ({
  activeTab: 'dashboard',
  selectedDataset: null,
  targetCol: '',
  taskType: '',
  excludedColsDraft: [],
  excludedColsApplied: [],
  filtersDraft: [],
  filtersApplied: [],
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

  addFilterDraft: (filter) =>
    set((state) => ({ filtersDraft: [...state.filtersDraft, filter] })),
  removeFilterDraft: (index) =>
    set((state) => ({ filtersDraft: state.filtersDraft.filter((_, i) => i !== index) })),
  clearFiltersDraft: () => set({ filtersDraft: [] }),
  setFiltersDraft: (filters) => set({ filtersDraft: filters }),
  setFiltersApplied: (filters) => set({ filtersApplied: filters }),
  applyFilters: () => set((state) => ({ filtersApplied: [...state.filtersDraft] })),

  setScatter: (patch) => set((state) => ({ scatter: { ...state.scatter, ...patch } })),

  resetForDataset: () =>
    set({
      excludedColsDraft: [],
      excludedColsApplied: [],
      filtersDraft: [],
      filtersApplied: [],
      targetCol: '',
      activeTab: 'dashboard',
      scatter: { ...EMPTY_SCATTER },
    }),
}));

/**
 * Selector — the draft and applied lists differ. EDAPage uses this to
 * decide whether to enable the "Apply Filters" button.
 */
export const selectExcludedDirty = (state: EDAState): boolean => {
  if (state.excludedColsApplied.length !== state.excludedColsDraft.length) return true;
  const applied = new Set(state.excludedColsApplied);
  for (const col of state.excludedColsDraft) {
    if (!applied.has(col)) return true;
  }
  return false;
};

/** Selector — filters are dirty whenever the draft differs from the applied snapshot. */
export const selectFiltersDirty = (state: EDAState): boolean =>
  !areFiltersEqual(state.filtersDraft, state.filtersApplied);
