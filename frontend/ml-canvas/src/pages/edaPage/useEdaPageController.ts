import { useMemo, useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { EDAService } from '../../core/api/eda';
import {
  useEDAStore, selectExcludedDirty, selectFiltersDirty, type EDAFilter,
} from '../../core/store/useEDAStore';
import type { ColumnProfile } from '../../core/types/edaProfile';
import { edaKeys } from '../../core/hooks/useEdaJobs';
import { resolveEdaDatasetSelection, shouldSyncDatasetParam, isSelectionMissingFromDatasets } from '../../core/utils/edaDatasetSelection';
import { toast } from '../../core/toast';
import { useEdaReports } from './useEdaReports';

/** The page owns these hooks unconditionally across report status and tab changes. */
export function useEdaPageController() {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // Only UI-toggle state remains local; server data lives in React Query, view state in the slice.
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [isLoadingReport, setIsLoadingReport] = useState(false);
  const [isApplyingFilters, setIsApplyingFilters] = useState(false);
  const applyingFiltersRef = useRef(false);

  // ── View + analysis-input state lives in the EDA zustand slice ──
  const activeTab = useEDAStore((s) => s.activeTab);
  const setActiveTab = useEDAStore((s) => s.setActiveTab);
  const selectedDataset = useEDAStore((s) => s.selectedDataset);
  const setSelectedDataset = useEDAStore((s) => s.setSelectedDataset);
  const targetCol = useEDAStore((s) => s.targetCol);
  const setTargetCol = useEDAStore((s) => s.setTargetCol);
  const taskType = useEDAStore((s) => s.taskType);
  const setTaskType = useEDAStore((s) => s.setTaskType);
  const excludedColsDraft = useEDAStore((s) => s.excludedColsDraft);
  const excludedColsApplied = useEDAStore((s) => s.excludedColsApplied);
  const filtersDraft = useEDAStore((s) => s.filtersDraft);
  const filtersApplied = useEDAStore((s) => s.filtersApplied);
  const scatter = useEDAStore((s) => s.scatter);
  const setScatter = useEDAStore((s) => s.setScatter);
  const excludedDirty = useEDAStore(selectExcludedDirty);
  const filtersDirty = useEDAStore(selectFiltersDirty);

  // ── React Query: datasets / latest report / history ──
  const {
    datasetsQuery, datasets, datasetOptions, reportQuery, report, loading, error, history,
  } = useEdaReports(selectedDataset);

  const analyzeMutation = useMutation({
    mutationFn: (params: { excluded: string[]; filters: EDAFilter[]; }) =>
      EDAService.analyze(
        selectedDataset!,
        targetCol || undefined,
        params.excluded,
        params.filters,
        taskType || undefined,
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: edaKeys.report(selectedDataset ?? null) });
      queryClient.invalidateQueries({ queryKey: edaKeys.history(selectedDataset ?? null) });
    },
    onSettled: () => {
      applyingFiltersRef.current = false;
      setIsApplyingFilters(false);
    },
  });
  const analyzing = analyzeMutation.isPending || isApplyingFilters;

  // The URL owns the dataset selection: an explicit `?dataset_id=` always wins,
  // so deep links and back/forward can't be silently overridden by the store
  // (which is a module singleton and survives page unmount).
  const datasetParam = searchParams.get('dataset_id');
  useEffect(() => {
    const resolved = resolveEdaDatasetSelection(datasetParam, selectedDataset, datasets);
    if (resolved !== selectedDataset) {
      setSelectedDataset(resolved);
    }
  }, [datasetParam, datasets, selectedDataset, setSelectedDataset]);

  // Mirror the resolved selection back into the URL so a reload or share
  // reopens the dataset the user is actually looking at.
  useEffect(() => {
    if (!shouldSyncDatasetParam(datasetParam, selectedDataset)) return;
    const next = new URLSearchParams(searchParams);
    next.set('dataset_id', String(selectedDataset));
    setSearchParams(next, { replace: true });
  }, [datasetParam, selectedDataset, searchParams, setSearchParams]);

  const selectionUnavailable = isSelectionMissingFromDatasets(
    selectedDataset,
    datasets,
    datasetsQuery.isSuccess,
  );

  // Wipe per-dataset slice fields whenever the user switches datasets.
  useEffect(() => {
    useEDAStore.getState().resetForDataset();
  }, [selectedDataset]);

  // Strip excluded columns from the profile for the UI without mutating the cached payload.
  const profileForUi = useMemo(() => {
    const rawProfile = report?.profile_data;
    if (!rawProfile) return null;

    const excludedSet = new Set(excludedColsDraft);
    const filteredColumns: Record<string, ColumnProfile> = {};
    if (rawProfile.columns) {
      Object.entries(rawProfile.columns).forEach(([name, col]) => {
        if (!excludedSet.has(name)) {
          filteredColumns[name] = col as ColumnProfile;
        }
      });
    }

    return {
      ...rawProfile,
      columns: filteredColumns,
      excluded_columns: excludedColsDraft,
    } as typeof rawProfile;
  }, [report?.profile_data, excludedColsDraft]);

  const runAnalysis = (overrideExcluded?: string[], overrideFilters?: EDAFilter[]) => {
    if (!selectedDataset) return;
    const actualExcluded = Array.isArray(overrideExcluded) ? overrideExcluded : excludedColsApplied;
    const actualFilters = Array.isArray(overrideFilters) ? overrideFilters : filtersApplied;
    analyzeMutation.mutate({ excluded: actualExcluded, filters: actualFilters });
  };

  // Load a non-latest report into the latest-cache slot so the existing UI renders it.
  const loadSpecificReport = async (reportId: number) => {
    if (!selectedDataset) return;
    setIsLoadingReport(true);
    try {
      const data = await queryClient.fetchQuery({
        queryKey: edaKeys.reportById(reportId),
        queryFn: () => EDAService.getReport(reportId),
      });
      queryClient.setQueryData(edaKeys.report(selectedDataset), data);
    } catch (error) {
      toast.error('Failed to load report', String(error));
    } finally {
      setIsLoadingReport(false);
    }
  };

  const handleAddFilter = (
    column: string,
    value: string | number | boolean | Array<string | number>,
    operator: string,
  ) => {
    const newFilter: EDAFilter = {
      column,
      operator: (operator || '==') as EDAFilter['operator'],
      value,
    };
    useEDAStore.getState().addFilterDraft(newFilter);
  };

  const handleRemoveFilter = (index: number) => {
    useEDAStore.getState().removeFilterDraft(index);
  };

  const handleResetFilters = () => {
    useEDAStore.getState().setFiltersDraft(filtersApplied);
  };

  const handleApplyFilters = () => {
    if (applyingFiltersRef.current || analyzing) return;
    applyingFiltersRef.current = true;
    setIsApplyingFilters(true);
    const draft = useEDAStore.getState().filtersDraft;
    useEDAStore.getState().applyFilters();
    runAnalysis(undefined, draft);
  };

  const handleToggleExclude = (colName: string, exclude: boolean) => {
    useEDAStore.getState().toggleExclude(colName, exclude);
  };

  const handleApplyExcluded = () => {
    const draft = useEDAStore.getState().excludedColsDraft;
    useEDAStore.getState().applyExcluded();
    runAnalysis(draft);
  };

  // Sync target col and excluded cols from report if available
  useEffect(() => {
    if (report && report.profile_data) {
      if (report.profile_data.target_col) {
        setTargetCol(report.profile_data.target_col);
      }
      const serverExcluded = Array.isArray(report.profile_data.excluded_columns)
        ? report.profile_data.excluded_columns
        : [];
      useEDAStore.getState().setExcludedApplied(serverExcluded);
      useEDAStore.getState().setExcludedDraft(serverExcluded);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report?.id]);

  // Helper to find existing report for current target
  const existingReport = targetCol ? history.find(h => h.target_col === targetCol && h.status === 'COMPLETED') : null;

  return {
    searchParams,
    setSearchParams,
    queryClient,
    showHistoryModal,
    setShowHistoryModal,
    isLoadingReport,
    activeTab,
    setActiveTab,
    selectedDataset,
    targetCol,
    setTargetCol,
    taskType,
    setTaskType,
    excludedColsDraft,
    filtersDraft,
    filtersApplied,
    scatter,
    setScatter,
    excludedDirty,
    filtersDirty,
    datasetOptions,
    reportQuery,
    report,
    loading,
    error,
    history,
    analyzeMutation,
    analyzing,
    selectionUnavailable,
    profileForUi,
    runAnalysis,
    loadSpecificReport,
    handleAddFilter,
    handleRemoveFilter,
    handleResetFilters,
    handleApplyFilters,
    handleToggleExclude,
    handleApplyExcluded,
    existingReport
  };
}

export type EdaPageModel = ReturnType<typeof useEdaPageController>;
