import React, { useMemo, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { DatasetService } from '../core/api/datasets';
import { EDAService } from '../core/api/eda';
import { JobsHistoryModal } from '../components/eda/JobsHistoryModal';
import { EDASidebar } from '../components/eda/EDASidebar';
import { DashboardTab } from '../components/eda/tabs/DashboardTab';
import { InsightsTab } from '../components/eda/tabs/InsightsTab';
import { PCATab } from '../components/eda/tabs/PCATab';
import { GeospatialTab } from '../components/eda/tabs/GeospatialTab';
import { TargetAnalysisTab } from '../components/eda/tabs/TargetAnalysisTab';
import { TimeSeriesTab } from '../components/eda/tabs/TimeSeriesTab';
import { VariablesTab } from '../components/eda/tabs/VariablesTab';
import { BivariateTab } from '../components/eda/tabs/BivariateTab';
import { OutliersTab } from '../components/eda/tabs/OutliersTab';
import { CorrelationsTab } from '../components/eda/tabs/CorrelationsTab';
import { SampleDataTab } from '../components/eda/tabs/SampleDataTab';
import { CausalTab } from '../components/eda/tabs/CausalTab';
import { RuleDiscoveryTab } from '../components/eda/tabs/RuleDiscoveryTab';
import { DecompositionTab } from '../components/eda/tabs/DecompositionTab';
import { Loader2, RefreshCw, AlertCircle, BarChart2, List, Play, HelpCircle, Target } from 'lucide-react';
import { downloadChart } from '../core/utils/chartUtils';
import { LoadingState, ErrorState } from '../components/shared';
import { useEDAStore, selectExcludedDirty, type EDAFilter } from '../core/store/useEDAStore';
import type { ColumnProfile } from '../core/types/edaProfile';
import { edaKeys } from '../core/hooks/useEdaJobs';

export const EDAPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const queryClient = useQueryClient();

  // Only UI-toggle state remains local; server data lives in React Query, view state in the slice.
  const [showHistoryModal, setShowHistoryModal] = useState(false);

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
  const filters = useEDAStore((s) => s.filters);
  const scatter = useEDAStore((s) => s.scatter);
  const setScatter = useEDAStore((s) => s.setScatter);
  const excludedDirty = useEDAStore(selectExcludedDirty);

  // Seed the dataset id from `?dataset_id=…` once on mount.
  useEffect(() => {
    const id = searchParams.get('dataset_id');
    if (id && useEDAStore.getState().selectedDataset == null) {
      setSelectedDataset(Number(id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── React Query: datasets / latest report / history ──
  const datasetsQuery = useQuery({
    queryKey: edaKeys.datasets,
    queryFn: () => DatasetService.getUsable(),
  });
  // Memoize the fallback so dependent effects don't refire on every render.
  const datasets = useMemo(() => datasetsQuery.data ?? [], [datasetsQuery.data]);

  const reportQuery = useQuery({
    queryKey: edaKeys.report(selectedDataset ?? null),
    queryFn: async () => {
      try {
        return await EDAService.getLatestReport(selectedDataset!);
      } catch (err: unknown) {
        // 404 simply means "no report yet" — surface as null rather than a hard error.
        const status = (err as { response?: { status?: number } })?.response?.status;
        if (status === 404) return null;
        throw err;
      }
    },
    enabled: selectedDataset != null,
    // Auto-poll every 3 s while the backend job is PENDING; stop once it completes/fails.
    refetchInterval: (query) => {
      const data = query.state.data;
      return data && data.status === 'PENDING' ? 3000 : false;
    },
  });
  const report = reportQuery.data ?? null;
  const loading = reportQuery.isLoading;
  const error = reportQuery.isError ? 'Failed to load report' : null;

  const historyQuery = useQuery({
    queryKey: edaKeys.history(selectedDataset ?? null),
    queryFn: () => EDAService.getHistory(selectedDataset!),
    enabled: selectedDataset != null,
    // Refresh history alongside the report while a job is in flight.
    refetchInterval: report?.status === 'PENDING' ? 3000 : false,
  });
  const history = historyQuery.data ?? [];

  // ── Mutations ──
  const analyzeMutation = useMutation({
    mutationFn: (params: { excluded: string[]; filters: EDAFilter[] }) =>
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
  });
  const analyzing = analyzeMutation.isPending;

  // Default-select the first dataset once the list loads.
  useEffect(() => {
    if (!selectedDataset && datasets.length > 0) {
      setSelectedDataset(Number(datasets[0]!.id));
    }
  }, [datasets, selectedDataset, setSelectedDataset]);

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
    const actualFilters = Array.isArray(overrideFilters) ? overrideFilters : filters;
    analyzeMutation.mutate({ excluded: actualExcluded, filters: actualFilters });
  };

  // Load a non-latest report into the latest-cache slot so the existing UI renders it.
  const loadSpecificReport = async (reportId: number) => {
    if (!selectedDataset) return;
    const data = await queryClient.fetchQuery({
      queryKey: edaKeys.reportById(reportId),
      queryFn: () => EDAService.getReport(reportId),
    });
    queryClient.setQueryData(edaKeys.report(selectedDataset), data);
  };

  const handleAddFilter = (column: string, value: string | number | boolean | Array<string | number>, operator: string) => {
      const newFilter: EDAFilter = { column, operator: (operator || '==') as EDAFilter['operator'], value };
      const newFilters = [...filters, newFilter];
      useEDAStore.getState().setFilters(newFilters);
      runAnalysis(undefined, newFilters);
  };

  const handleRemoveFilter = (index: number) => {
      const newFilters = filters.filter((_, i) => i !== index);
      useEDAStore.getState().setFilters(newFilters);
      runAnalysis(undefined, newFilters);
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

  const renderContent = () => {
    if (loading && !report) {
      return <LoadingState message="Analyzing dataset..." />;
    }

    if (error) {
      return (
        <ErrorState
          error={error}
          onRetry={() => reportQuery.refetch()}
        />
      );
    }

    if (!report) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <BarChart2 className="w-16 h-16 mb-4 opacity-20" />
          <p className="mb-4">No analysis found for this dataset.</p>
          
          <div className="flex flex-col items-center space-y-4">
            <div className="w-64 space-y-2">
                <input
                type="text"
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="Target Column (Optional)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
                
                <div className="flex items-center space-x-2">
                    <select
                        value={taskType}
                        onChange={(e) => setTaskType(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    >
                        <option value="">Auto-Detect Task</option>
                        <option value="Classification">Classification</option>
                        <option value="Regression">Regression</option>
                    </select>
                    <div className="group relative flex items-center">
                        <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
                        <div className="absolute left-full ml-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                            Force a specific task type. Useful for ID columns (force Classification) or numeric categories (force Regression).
                        </div>
                    </div>
                </div>
            </div>
            <div className="flex gap-2">
                {existingReport && (
                    <button
                        onClick={() => loadSpecificReport(existingReport.id)}
                        className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Load Existing ({new Date(existingReport.created_at).toLocaleDateString()})
                    </button>
                )}
                <button
                    onClick={() => runAnalysis()}
                    disabled={analyzing}
                    className={`flex items-center px-4 py-2 ${existingReport ? 'bg-gray-600 hover:bg-gray-700' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded-md disabled:opacity-50`}
                >
                    {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                    {existingReport ? 'Run New Analysis' : 'Run Analysis'}
                </button>
            </div>
          </div>
        </div>
      );
    }

    if (report.status === 'PENDING') {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500">
          <Loader2 className="w-16 h-16 mb-4 animate-spin text-blue-500" />
          <p>Analysis in progress...</p>
          <p className="text-sm text-gray-400">This may take a few moments.</p>
        </div>
      );
    }

    if (report.status === 'FAILED') {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-red-500">
          <AlertCircle className="w-16 h-16 mb-4" />
          <p>Analysis Failed</p>
          <p className="text-sm text-gray-600 mt-2">{report.error_message}</p>
          <button
            onClick={() => runAnalysis()}
            className="mt-4 flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </button>
        </div>
      );
    }

    const profile = profileForUi;
    if (!profile) return <div>No profile data</div>;

    const allColumns = report?.profile_data?.columns ? Object.keys(report.profile_data.columns) : [];

    return (
      <div className="flex h-full w-full overflow-hidden bg-white dark:bg-gray-900">
        <EDASidebar 
            activeTab={activeTab} 
            setActiveTab={setActiveTab} 
            profile={profile} 
            filters={filters}
          columns={allColumns}
            excludedCols={excludedColsDraft}
            excludedDirty={excludedDirty}
            analyzing={analyzing}
            onAddFilter={handleAddFilter}
            onRemoveFilter={handleRemoveFilter}
            onClearFilters={() => { useEDAStore.getState().clearFilters(); runAnalysis(undefined, []); }}
            onToggleExclude={handleToggleExclude}
            onApplyExcluded={handleApplyExcluded}
        />
        
        <div className="flex-1 overflow-y-auto p-6 pt-4 bg-gray-50 dark:bg-gray-900/50">
            {activeTab === 'dashboard' && (
                <DashboardTab 
                    profile={profile} 
                    onToggleExclude={handleToggleExclude}
                    excludedCols={excludedColsDraft}
                />
            )}

            {activeTab === 'insights' && (
                <InsightsTab profile={profile} />
            )}

            {activeTab === 'pca' && (
                <PCATab 
                    profile={profile} 
                    isPCA3D={scatter.isPCA3D} 
                    setIsPCA3D={(v) => setScatter({ isPCA3D: v })} 
                    downloadChart={downloadChart} 
                />
            )}

            {activeTab === 'geospatial' && !!profile.geospatial && (
                <GeospatialTab profile={profile} />
            )}

            {activeTab === 'target' && !!profile.target_col && !!profile.target_correlations && (
                <TargetAnalysisTab 
                    profile={profile}
                    downloadChart={downloadChart}
                    history={history}
                    loading={loading}
                    loadSpecificReport={loadSpecificReport}
                    report={report}
                />
            )}

            {activeTab === 'timeseries' && !!profile.timeseries && (
                <TimeSeriesTab 
                    profile={profile}
                    downloadChart={downloadChart}
                />
            )}

            {activeTab === 'variables' && (
                <VariablesTab 
                    profile={profile}
                    handleToggleExclude={handleToggleExclude}
                    handleAddFilter={handleAddFilter}
                />
            )}

            {activeTab === 'bivariate' && (
                <BivariateTab 
                    profile={profile}
                    downloadChart={downloadChart}
                    scatterX={scatter.x}
                    setScatterX={(v) => setScatter({ x: v })}
                    scatterY={scatter.y}
                    setScatterY={(v) => setScatter({ y: v })}
                    scatterZ={scatter.z}
                    setScatterZ={(v) => setScatter({ z: v })}
                    scatterColor={scatter.color}
                    setScatterColor={(v) => setScatter({ color: v })}
                    is3D={scatter.is3D}
                    setIs3D={(v) => setScatter({ is3D: v })}
                />
            )}

            {activeTab === 'outliers' && profile.outliers && (
                <OutliersTab profile={profile} />
            )}

            {activeTab === 'correlations' && (!!profile.correlations || !!profile.correlations_with_target) && (
                <CorrelationsTab 
                    profile={profile}
                />
            )}

            {activeTab === 'causal' && !!profile.causal_graph && (
                <CausalTab profile={profile} />
            )}

            {activeTab === 'rules' && !!profile.rule_tree && (
                <RuleDiscoveryTab profile={profile} />
            )}

            {activeTab === 'decomposition' && selectedDataset && (
                <DecompositionTab 
                    datasetId={selectedDataset}
                columns={allColumns}
                    initialFilters={filters}
                />
            )}

            {activeTab === 'sample' && profile.sample_data && (
                <SampleDataTab 
                    profile={profile}
                excludedCols={excludedColsDraft}
                    handleToggleExclude={handleToggleExclude}
                />
            )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full w-full overflow-hidden bg-white dark:bg-slate-950">
      {/* Top Navigation Bar */}
      <header className="flex-none h-16 bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-gray-800 px-4 flex items-center justify-between gap-4 z-20 shadow-sm">
        
        {/* Left: Title & Dataset */}
        <div className="flex items-center gap-6">
            <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white leading-tight">Exploratory Analysis</h1>
                {report && report.created_at && (
                    <p className="text-[10px] text-gray-500">
                        Last analyzed: {new Date(report.created_at).toLocaleString()}
                    </p>
                )}
            </div>

            <div className="h-8 w-px bg-gray-200 dark:bg-gray-700 mx-2"></div>
            
            <div className="flex flex-col">
                <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Dataset</span>
                <select
                    value={selectedDataset || ''}
                    onChange={(e) => setSelectedDataset(Number(e.target.value))}
                    className="block w-48 text-sm font-medium bg-transparent border-none p-0 focus:ring-0 text-gray-900 dark:text-white cursor-pointer hover:text-blue-600"
                >
                    <option value="" disabled>Select a dataset</option>
                    {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>{ds.name}</option>
                    ))}
                </select>
            </div>
        </div>

        {/* Center: Controls */}
        <div className="flex items-center gap-4 bg-gray-50 dark:bg-gray-800/50 p-1.5 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex flex-col px-2">
                <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Target Column</span>
                <select
                    value={targetCol}
                    onChange={(e) => setTargetCol(e.target.value)}
                    disabled={!report || !report.profile_data}
                    className="block w-36 text-sm bg-transparent border-none p-0 focus:ring-0 text-gray-700 dark:text-gray-200 cursor-pointer disabled:opacity-50"
                >
                    <option value="">None</option>
                    {report && report.profile_data && Object.keys(report.profile_data.columns)
                        .filter((col) => !excludedColsDraft.includes(col))
                        .map(col => (
                            <option key={col} value={col}>{col}</option>
                    ))}
                </select>
            </div>

            <div className="w-px h-8 bg-gray-200 dark:bg-gray-700"></div>

            <div className="flex flex-col px-2">
                <div className="flex items-center gap-1">
                    <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Task Type</span>
                    <div className="group relative">
                        <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
                        <div className="absolute bottom-full mb-2 w-56 p-2 bg-slate-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none left-1/2 -translate-x-1/2">
                            Force Classification or Regression.
                        </div>
                    </div>
                </div>
                <select
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                    disabled={!report || !report.profile_data}
                    className="block w-28 text-sm bg-transparent border-none p-0 focus:ring-0 text-gray-700 dark:text-gray-200 cursor-pointer disabled:opacity-50"
                >
                    <option value="">Auto</option>
                    <option value="Classification">Classification</option>
                    <option value="Regression">Regression</option>
                </select>
            </div>

            <button
                onClick={() => selectedDataset && runAnalysis()}
                disabled={!selectedDataset || analyzing}
                className={`ml-2 px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center shadow-sm ${
                    existingReport 
                        ? 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50' 
                        : 'bg-blue-600 text-white hover:bg-blue-700 border border-transparent'
                }`}
            >
                {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                {existingReport ? 'Re-Run' : 'Analyze'}
            </button>
        </div>

        {/* Right: History & Actions */}
        <div className="flex items-center gap-3">
             {existingReport && report && report.id !== existingReport.id && (
                <button
                    onClick={() => loadSpecificReport(existingReport.id)}
                    className="flex items-center px-3 py-1.5 text-xs bg-green-50 text-green-700 border border-green-200 rounded-md hover:bg-green-100 transition-colors"
                    title={`Load existing from ${new Date(existingReport.created_at).toLocaleString()}`}
                >
                    <RefreshCw className="w-3 h-3 mr-1" />
                    Load Saved
                </button>
            )}
            
            <button 
                onClick={() => setShowHistoryModal(true)}
                className="flex items-center px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-50 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
                <List className="w-4 h-4 mr-2" />
                History
            </button>
        </div>
      </header>

      {/* Recent Targets Bar */}
      {history.length > 0 && (
        <div className="flex-none bg-white dark:bg-slate-950 border-b border-gray-100 dark:border-gray-800 px-4 py-1.5 flex items-center gap-3 overflow-x-auto z-10 shadow-[0_2px_3px_-1px_rgba(0,0,0,0.02)]">
            <span className="text-[10px] uppercase font-bold text-gray-400 tracking-wider whitespace-nowrap">Recent Targets:</span>
            <div className="flex gap-2">
                {Array.from(new Set(history.filter(h => h.target_col && h.status === 'COMPLETED').map(h => h.target_col))).slice(0, 8).map(target => (
                    <button
                        key={target}
                        onClick={() => {
                            const match = history.find(h => h.target_col === target && h.status === 'COMPLETED');
                            if (match) loadSpecificReport(match.id);
                        }}
                        className={`px-2 py-0.5 text-xs rounded-full border transition-colors flex items-center ${
                            report?.profile_data?.target_col === target 
                            ? 'bg-blue-50 text-blue-600 border-blue-200 font-medium'
                            : 'bg-white text-gray-500 border-gray-200 hover:border-gray-300 hover:text-gray-700'
                        }`}
                    >
                        <Target className="w-3 h-3 mr-1" />
                        {target}
                    </button>
                ))}
            </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden relative">
        {renderContent()}
      </div>

      {/* Jobs History Modal */}
      <JobsHistoryModal 
        isOpen={showHistoryModal}
        onClose={() => setShowHistoryModal(false)}
        history={history}
        datasetId={selectedDataset ?? null}
        onRefresh={() =>
          queryClient.invalidateQueries({ queryKey: edaKeys.history(selectedDataset ?? null) })
        }
        onFetchReport={async (id) => {
            const r = await EDAService.getReport(id);
            // EDAReport.id is optional in the API type; here we know the route
            // returned a real report so the id is always present.
            return { ...r, id: r.id ?? id };
        }}
        onSelect={(selectedReport) => {
            if (selectedDataset) {
              queryClient.setQueryData(edaKeys.report(selectedDataset), selectedReport);
            }
          const serverExcluded = Array.isArray(selectedReport.profile_data?.excluded_columns)
            ? selectedReport.profile_data.excluded_columns
            : [];
          useEDAStore.getState().setExcludedApplied(serverExcluded);
          useEDAStore.getState().setExcludedDraft(serverExcluded);
        }}
      />
    </div>
  );
};
