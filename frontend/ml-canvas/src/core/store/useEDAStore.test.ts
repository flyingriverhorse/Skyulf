import { describe, it, expect, beforeEach } from 'vitest';
import { useEDAStore, selectExcludedDirty, type EDAFilter } from './useEDAStore';

const reset = () =>
  useEDAStore.setState({
    activeTab: 'dashboard',
    selectedDataset: null,
    targetCol: '',
    taskType: '',
    excludedColsDraft: [],
    excludedColsApplied: [],
    filters: [],
    scatter: { x: '', y: '', z: '', color: '', is3D: false, isPCA3D: false },
  });

describe('useEDAStore — basic setters', () => {
  beforeEach(reset);

  it('updates activeTab / selectedDataset / target / task', () => {
    const s = useEDAStore.getState();
    s.setActiveTab('variables');
    s.setSelectedDataset(7);
    s.setTargetCol('price');
    s.setTaskType('Regression');

    const next = useEDAStore.getState();
    expect(next.activeTab).toBe('variables');
    expect(next.selectedDataset).toBe(7);
    expect(next.targetCol).toBe('price');
    expect(next.taskType).toBe('Regression');
  });
});

describe('useEDAStore — exclude draft/applied workflow', () => {
  beforeEach(reset);

  it('toggleExclude adds and removes columns idempotently', () => {
    const { toggleExclude } = useEDAStore.getState();
    toggleExclude('age', true);
    toggleExclude('age', true); // second add is a no-op
    expect(useEDAStore.getState().excludedColsDraft).toEqual(['age']);

    toggleExclude('income', true);
    expect(useEDAStore.getState().excludedColsDraft).toEqual(['age', 'income']);

    toggleExclude('age', false);
    toggleExclude('age', false); // second remove is a no-op
    expect(useEDAStore.getState().excludedColsDraft).toEqual(['income']);
  });

  it('applyExcluded copies draft to applied (snapshot, not reference)', () => {
    const { toggleExclude, applyExcluded } = useEDAStore.getState();
    toggleExclude('a', true);
    toggleExclude('b', true);
    applyExcluded();

    const { excludedColsDraft, excludedColsApplied } = useEDAStore.getState();
    expect(excludedColsApplied).toEqual(['a', 'b']);
    expect(excludedColsApplied).not.toBe(excludedColsDraft);

    // Subsequent draft edits do not leak into applied.
    useEDAStore.getState().toggleExclude('c', true);
    expect(useEDAStore.getState().excludedColsApplied).toEqual(['a', 'b']);
  });

  it('selectExcludedDirty true when sets differ, false when equal', () => {
    const { toggleExclude, applyExcluded, setExcludedDraft } = useEDAStore.getState();
    expect(selectExcludedDirty(useEDAStore.getState())).toBe(false);

    toggleExclude('a', true);
    expect(selectExcludedDirty(useEDAStore.getState())).toBe(true);

    applyExcluded();
    expect(selectExcludedDirty(useEDAStore.getState())).toBe(false);

    // Same length, different contents → still dirty.
    setExcludedDraft(['z']);
    expect(selectExcludedDirty(useEDAStore.getState())).toBe(true);
  });
});

describe('useEDAStore — filters', () => {
  beforeEach(reset);

  it('addFilter / removeFilter / clearFilters', () => {
    const f1: EDAFilter = { column: 'age', operator: '>=', value: 18 };
    const f2: EDAFilter = { column: 'city', operator: '==', value: 'NY' };

    const { addFilter, removeFilter, clearFilters } = useEDAStore.getState();
    addFilter(f1);
    addFilter(f2);
    expect(useEDAStore.getState().filters).toEqual([f1, f2]);

    removeFilter(0);
    expect(useEDAStore.getState().filters).toEqual([f2]);

    clearFilters();
    expect(useEDAStore.getState().filters).toEqual([]);
  });
});

describe('useEDAStore — scatter axes', () => {
  beforeEach(reset);

  it('setScatter merges patches without clobbering siblings', () => {
    useEDAStore.getState().setScatter({ x: 'a', y: 'b' });
    useEDAStore.getState().setScatter({ z: 'c', is3D: true });
    expect(useEDAStore.getState().scatter).toEqual({
      x: 'a',
      y: 'b',
      z: 'c',
      color: '',
      is3D: true,
      isPCA3D: false,
    });
  });
});

describe('useEDAStore — resetForDataset', () => {
  beforeEach(reset);

  it('clears per-dataset state but keeps `selectedDataset` (the page sets that separately)', () => {
    const s = useEDAStore.getState();
    s.setSelectedDataset(42);
    s.setTargetCol('y');
    s.toggleExclude('a', true);
    s.applyExcluded();
    s.addFilter({ column: 'x', operator: '>', value: 0 });
    s.setScatter({ x: 'lat', y: 'lon', is3D: true });
    s.setActiveTab('variables');

    s.resetForDataset();

    const next = useEDAStore.getState();
    expect(next.selectedDataset).toBe(42); // intentionally preserved
    expect(next.targetCol).toBe('');
    expect(next.excludedColsDraft).toEqual([]);
    expect(next.excludedColsApplied).toEqual([]);
    expect(next.filters).toEqual([]);
    expect(next.scatter).toEqual({
      x: '',
      y: '',
      z: '',
      color: '',
      is3D: false,
      isPCA3D: false,
    });
    expect(next.activeTab).toBe('dashboard');
  });
});
