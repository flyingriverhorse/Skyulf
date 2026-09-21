import type { PropsWithChildren } from 'react';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { EDAReport, Filter } from '../../core/api/eda';
import { edaKeys } from '../../core/hooks/useEdaJobs';
import { useEDAStore } from '../../core/store/useEDAStore';
import { useEdaPageController } from './useEdaPageController';

const savedFilters: Filter[] = [{ column: 'region', operator: 'in', value: ['North', 'South'] }];
const profile = { row_count: 3, column_count: 0, columns: {}, excluded_columns: ['unused'] };

/** Mount the real controller against cached server responses without network requests. */
function mountController(report: EDAReport) {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } });
  client.setQueryData(edaKeys.datasets, [{ id: '1', name: 'First' }, { id: '2', name: 'Second' }]);
  client.setQueryData(edaKeys.report(1), report);
  client.setQueryData(edaKeys.history(1), []);
  client.setQueryData(edaKeys.history(2), []);
  const wrapper = ({ children }: PropsWithChildren) => (
    <MemoryRouter initialEntries={['/?dataset_id=1']}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
  return { ...renderHook(useEdaPageController, { wrapper }), client };
}

describe('saved EDA report filters', () => {
  it('adds a histogram range atomically while preserving existing and applied filters', async () => {
    // Store subscribers must never observe only one half of the selected range.
    const { result } = mountController({ id: 11, status: 'COMPLETED', profile_data: profile, config: { filters: savedFilters } });
    await waitFor(() => expect(result.current.filtersApplied).toEqual(savedFilters));
    const range: Filter[] = [
      { column: 'price', operator: '>', value: 1 }, { column: 'price', operator: '<=', value: 2 },
    ];
    const observed: Filter[][] = [];
    const unsubscribe = useEDAStore.subscribe(state => observed.push(state.filtersDraft));
    try {
      act(() => result.current.handleAddFilters(range));
    } finally {
      unsubscribe();
    }
    expect(observed).toEqual([[...savedFilters, ...range]]);
    expect(result.current.filtersApplied).toEqual(savedFilters);
    expect(result.current.filtersDirty).toBe(true);
  });

  beforeEach(() => useEDAStore.setState(useEDAStore.getInitialState()));
  afterEach(cleanup);

  it('restores saved filters on page load and resets edits to that applied snapshot', async () => {
    /** Reset must discard only unsaved edits, retaining the report cohort. */
    const { result, client } = mountController({ id: 11, status: 'COMPLETED', profile_data: profile, config: { filters: savedFilters } });
    await waitFor(() => expect(result.current.filtersApplied).toEqual(savedFilters));
    expect(result.current.filtersDraft).toEqual(savedFilters);
    act(() => result.current.handleAddFilter('price', 5, '>'));
    expect(result.current.filtersDirty).toBe(true);
    act(() => client.setQueryData(edaKeys.report(1), {
      id: 11, status: 'COMPLETED', profile_data: { ...profile, row_count: 4 }, config: { filters: savedFilters },
    }));
    await waitFor(() => expect(result.current.report?.profile_data?.row_count).toBe(4));
    expect(result.current.filtersDraft).toHaveLength(2);
    act(() => result.current.handleResetFilters());
    expect(result.current.filtersDraft).toEqual(savedFilters);
    expect(result.current.filtersDirty).toBe(false);
  });

  it('replaces filters when opening history and clears them for an unfiltered dataset report', async () => {
    /** Historical and other-dataset views must never inherit a previous report cohort. */
    const { result, client } = mountController({ id: 11, status: 'COMPLETED', profile_data: profile, config: { filters: savedFilters } });
    const historicalFilters: Filter[] = [{ column: 'price', operator: '>', value: 10 }];
    client.setQueryData(edaKeys.reportById(10), {
      id: 10, status: 'COMPLETED', profile_data: profile, config: { filters: historicalFilters },
    });
    await act(() => result.current.loadSpecificReport(10));
    await waitFor(() => expect(result.current.filtersApplied).toEqual(historicalFilters));
    expect(result.current.filtersDraft).toEqual(historicalFilters);
    client.setQueryData(edaKeys.report(2), { id: 21, status: 'COMPLETED', profile_data: profile });
    act(() => result.current.setSearchParams({ dataset_id: '2' }));
    await waitFor(() => expect(result.current.selectedDataset).toBe(2));
    expect(result.current.filtersApplied).toEqual([]);
    expect(result.current.filtersDraft).toEqual([]);
    act(() => result.current.setSearchParams({ dataset_id: '1' }));
    await waitFor(() => expect(result.current.filtersApplied).toEqual(historicalFilters));
    expect(result.current.filtersDraft).toEqual(historicalFilters);
  });

  it('hydrates a completed profile when polling retains the pending report id', async () => {
    /** Completion must hydrate provenance even if the job id does not change. */
    const { result, client } = mountController({ id: 11, status: 'PENDING', config: { filters: savedFilters } });
    act(() => client.setQueryData(edaKeys.report(1), {
      id: 11, status: 'COMPLETED', config: { filters: savedFilters },
      profile_data: { ...profile, target_col: 'outcome' },
    }));
    await waitFor(() => expect(result.current.targetCol).toBe('outcome'));
    expect(result.current.excludedColsDraft).toEqual(['unused']);
    expect(result.current.filtersApplied).toEqual(savedFilters);
    expect(result.current.filtersDraft).toEqual(savedFilters);
  });
});
