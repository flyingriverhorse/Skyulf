import React, { useMemo, useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { EDAService } from '../core/api/eda';
import { JobsHistoryModal } from '../components/eda/JobsHistoryModal';
import { VariableDetailModal } from '../components/eda/VariableDetailModal';
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
import { ClusteringTab } from '../components/eda/tabs/ClusteringTab';
import { Loader2, RefreshCw, AlertCircle, BarChart2, List, Play, HelpCircle } from 'lucide-react';
import { downloadChart } from '../core/utils/chartUtils';

export const EDAPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<number | null>(() => {
    const id = searchParams.get('dataset_id');
    return id ? Number(id) : null;
  });
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [targetCol, setTargetCol] = useState<string>('');
  const [taskType, setTaskType] = useState<string>(''); // "Classification" or "Regression" or "" (Auto)
  const [excludedColsDraft, setExcludedColsDraft] = useState<string[]>([]);
  const [excludedColsApplied, setExcludedColsApplied] = useState<string[]>([]);
  const [filters, setFilters] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [selectedVariable, setSelectedVariable] = useState<any>(null);
  
  // Manual Filter State (Moved to FilterBar)

  // Scatter Plot State
  const [scatterX, setScatterX] = useState<string>('');
  const [scatterY, setScatterY] = useState<string>('');
  const [scatterZ, setScatterZ] = useState<string>('');
  const [scatterColor, setScatterColor] = useState<string>('');
  const [is3D, setIs3D] = useState(false);
  const [isPCA3D, setIsPCA3D] = useState(false);

  const profileForUi = useMemo(() => {
    const rawProfile = report?.profile_data;
    if (!rawProfile) return null;

    const excludedSet = new Set(excludedColsDraft);
    const filteredColumns: Record<string, any> = {};
    if (rawProfile.columns) {
      Object.entries(rawProfile.columns).forEach(([name, col]) => {
        if (!excludedSet.has(name)) {
          filteredColumns[name] = col;
        }
      });
    }

    return {
      ...rawProfile,
      columns: filteredColumns,
      excluded_columns: excludedColsDraft,
    };
  }, [report?.profile_data, excludedColsDraft]);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      setReport(null); // Clear previous report to show loader
      setExcludedColsDraft([]); // Reset excluded columns
      setExcludedColsApplied([]);
      setFilters([]); // Reset filters
      setTargetCol(''); // Reset target column
      setSelectedVariable(null); // Reset selected variable
      setScatterX('');
      setScatterY('');
      setScatterZ('');
      setScatterColor('');
      setActiveTab('dashboard'); // Reset active tab
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } else {
      setReport(null);
      setHistory([]);
      setExcludedColsDraft([]);
      setExcludedColsApplied([]);
      setFilters([]);
      setTargetCol('');
      setSelectedVariable(null);
      setScatterX('');
      setScatterY('');
      setScatterZ('');
      setScatterColor('');
      setActiveTab('dashboard');
    }
  }, [selectedDataset]);

  // Poll for status if pending
  useEffect(() => {
    let interval: any;
    if (report && report.status === 'PENDING') {
      interval = setInterval(() => {
        if (selectedDataset) {
            loadReport(selectedDataset, true);
            loadHistory(selectedDataset);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [report, selectedDataset]);

  const loadDatasets = async () => {
    try {
      const data = await DatasetService.getUsable();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(Number(data[0].id));
      }
    } catch (err) {
      console.error("Failed to load datasets", err);
    }
  };

  const loadHistory = async (id: number) => {
    try {
        const data = await EDAService.getHistory(id);
        setHistory(data);
    } catch (err) {
        console.error("Failed to load history", err);
    }
  };

  const loadReport = async (id: number, silent = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const data = await EDAService.getLatestReport(id);
      setReport(data);
    } catch (err: any) {
      if (err.response && err.response.status === 404) {
        setReport(null); // No report yet
      } else {
        // Only show error if not silent (polling)
        if (!silent) setError("Failed to load report");
      }
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const loadSpecificReport = async (reportId: number) => {
    setLoading(true);
    try {
        const data = await EDAService.getReport(reportId);
        setReport(data);
    } catch (err) {
        setError("Failed to load report");
    } finally {
        setLoading(false);
    }
  };

  const runAnalysis = async (overrideExcluded?: string[] | any, overrideFilters?: any[]) => {
    const actualExcluded = Array.isArray(overrideExcluded) ? overrideExcluded : excludedColsApplied;
    const actualFilters = Array.isArray(overrideFilters) ? overrideFilters : filters;

    if (!selectedDataset) return;
    setAnalyzing(true);
    try {
      await EDAService.analyze(selectedDataset, targetCol || undefined, actualExcluded, actualFilters, taskType || undefined);
      // Reload immediately to get the PENDING state
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } catch (err) {
      setError("Failed to start analysis");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleAddFilter = (column: string, value: any, operator: string = '==') => {
      // Check if filter already exists to avoid duplicates if needed, 
      // but for now let's allow multiple filters on same col (e.g. range)
      const newFilter = { column, operator, value };
      const newFilters = [...filters, newFilter];
      setFilters(newFilters);
      runAnalysis(undefined, newFilters);
  };

  const handleRemoveFilter = (index: number) => {
      const newFilters = [...filters];
      newFilters.splice(index, 1);
      setFilters(newFilters);
      runAnalysis(undefined, newFilters);
  };

  const handleToggleExclude = (colName: string, exclude: boolean) => {
    setExcludedColsDraft((prev) => {
      if (exclude) {
        if (prev.includes(colName)) return prev;
        return [...prev, colName];
      }
      return prev.filter((c) => c !== colName);
    });
  };

    const handleApplyExcluded = () => {
    setExcludedColsApplied(excludedColsDraft);
    runAnalysis(excludedColsDraft);
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
      setExcludedColsApplied(serverExcluded);
      setExcludedColsDraft(serverExcluded);
    }
    }, [report?.id]);

    const excludedDirty = (() => {
    if (excludedColsApplied.length !== excludedColsDraft.length) return true;
    const appliedSet = new Set(excludedColsApplied);
    for (const col of excludedColsDraft) {
      if (!appliedSet.has(col)) return true;
    }
    return false;
    })();

  // Helper to find existing report for current target
  const existingReport = targetCol ? history.find(h => h.target_col === targetCol && h.status === 'COMPLETED') : null;

  const renderContent = () => {
    if (loading && !report) {
      return (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-red-500">
          <AlertCircle className="w-16 h-16 mb-4" />
          <p>Error</p>
          <p className="text-sm text-gray-600 mt-2">{error}</p>
          <button
            onClick={() => selectedDataset && loadReport(selectedDataset)}
            className="mt-4 flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </button>
        </div>
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
      <div className="flex h-[calc(100vh-124px)] border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
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
            onClearFilters={() => { setFilters([]); runAnalysis(undefined, []); }}
            onToggleExclude={handleToggleExclude}
            onApplyExcluded={handleApplyExcluded}
        />
        
        <div className="flex-1 overflow-y-auto p-6 pt-4 bg-gray-50 dark:bg-gray-900/50">
            {activeTab === 'dashboard' && (
                <DashboardTab profile={profile} />
            )}

            {activeTab === 'insights' && (
                <InsightsTab profile={profile} />
            )}

            {activeTab === 'pca' && (
                <PCATab 
                    profile={profile} 
                    isPCA3D={isPCA3D} 
                    setIsPCA3D={setIsPCA3D} 
                    downloadChart={downloadChart} 
                />
            )}

            {activeTab === 'geospatial' && profile.geospatial && (
                <GeospatialTab profile={profile} />
            )}

            {activeTab === 'target' && profile.target_col && profile.target_correlations && (
                <TargetAnalysisTab 
                    profile={profile}
                    downloadChart={downloadChart}
                    history={history}
                    loading={loading}
                    loadSpecificReport={loadSpecificReport}
                    report={report}
                />
            )}

            {activeTab === 'timeseries' && profile.timeseries && (
                <TimeSeriesTab 
                    profile={profile}
                    downloadChart={downloadChart}
                />
            )}

            {activeTab === 'variables' && (
                <VariablesTab 
                    profile={profile}
                    setSelectedVariable={setSelectedVariable}
                    handleToggleExclude={handleToggleExclude}
                />
            )}

            {activeTab === 'bivariate' && (
                <BivariateTab 
                    profile={profile}
                    downloadChart={downloadChart}
                    scatterX={scatterX}
                    setScatterX={setScatterX}
                    scatterY={scatterY}
                    setScatterY={setScatterY}
                    scatterZ={scatterZ}
                    setScatterZ={setScatterZ}
                    scatterColor={scatterColor}
                    setScatterColor={setScatterColor}
                    is3D={is3D}
                    setIs3D={setIs3D}
                />
            )}

            {activeTab === 'outliers' && profile.outliers && (
                <OutliersTab profile={profile} />
            )}

            {activeTab === 'correlations' && (profile.correlations || profile.correlations_with_target) && (
                <CorrelationsTab 
                    profile={profile}
                />
            )}

            {activeTab === 'causal' && profile.causal_graph && (
                <CausalTab profile={profile} />
            )}

            {activeTab === 'rules' && profile.rule_tree && (
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
    <div id="eda-report-container" className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-4">
            <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory Data Analysis</h1>
                {report && report.created_at && (
                    <p className="text-xs text-gray-500 mt-1">
                        Last analyzed: {new Date(report.created_at).toLocaleString()}
                    </p>
                )}
            </div>
            <div className="flex gap-2 no-print">
                <button 
                    onClick={() => setShowHistoryModal(true)}
                    className="flex items-center px-3 py-1.5 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors ml-4"
                >
                    <List className="w-4 h-4 mr-2" />
                    Jobs History
                </button>
            </div>
        </div>
        <div className="flex items-center gap-4 no-print">
          <select
            value={selectedDataset || ''}
            onChange={(e) => setSelectedDataset(Number(e.target.value))}
            className="block w-48 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
          >
            <option value="" disabled>Select a dataset</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>{ds.name}</option>
            ))}
          </select>

          <div className="flex items-center gap-2">
             <span className="text-sm text-gray-500">Target:</span>
             <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                disabled={!report || !report.profile_data}
                className="block w-40 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border disabled:opacity-50"
             >
                <option value="">None</option>
               {report && report.profile_data && Object.keys(report.profile_data.columns)
                  .filter((col) => !excludedColsDraft.includes(col))
                  .map(col => (
                    <option key={col} value={col}>{col}</option>
                ))}
             </select>
             
             <div className="flex items-center gap-1">
                <select
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                    disabled={!report || !report.profile_data}
                    className="block w-32 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border disabled:opacity-50"
                    title="Force Task Type"
                >
                    <option value="">Auto</option>
                    <option value="Classification">Classif.</option>
                    <option value="Regression">Regress.</option>
                </select>
                <div className="group relative flex items-center">
                    <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
                    <div className="absolute right-0 top-full mt-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                        Force a specific task type. Useful for ID columns (force Classification) or numeric categories (force Regression).
                    </div>
                </div>
             </div>
          </div>

          <button 
            onClick={() => selectedDataset && runAnalysis()}
            disabled={!selectedDataset || analyzing}
            className="p-2 text-gray-500 hover:text-blue-600 rounded-full hover:bg-gray-100"
            title="Refresh Analysis"
          >
            <RefreshCw className={`w-5 h-5 ${analyzing ? 'animate-spin' : ''}`} />
          </button>
          
          {/* Quick Load Button if existing report found for selected target */}
          {existingReport && report && report.id !== existingReport.id && (
             <button
                onClick={() => loadSpecificReport(existingReport.id)}
                className="flex items-center px-3 py-1.5 text-xs bg-green-100 text-green-700 border border-green-200 rounded-md hover:bg-green-200 transition-colors"
                title={`Load existing analysis from ${new Date(existingReport.created_at).toLocaleString()}`}
             >
                <RefreshCw className="w-3 h-3 mr-1" />
                Load Existing
             </button>
          )}
        </div>
      </div>

      {/* Recent Targets Chips */}
      {history.length > 0 && (
        <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-2 no-print">
            <span className="text-xs text-gray-500 font-medium whitespace-nowrap">Recent Targets:</span>
            {Array.from(new Set(history.filter(h => h.target_col && h.status === 'COMPLETED').map(h => h.target_col))).slice(0, 5).map(target => (
                <button
                    key={target}
                    onClick={() => {
                        const match = history.find(h => h.target_col === target && h.status === 'COMPLETED');
                        if (match) loadSpecificReport(match.id);
                    }}
                    className={`px-2 py-1 text-xs rounded-full border transition-colors ${
                        report?.profile_data?.target_col === target 
                        ? 'bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800'
                        : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-700 dark:hover:bg-gray-700'
                    }`}
                >
                    {target}
                </button>
            ))}
        </div>
      )}

      {renderContent()}

      {/* Variable Detail Modal */}
      <VariableDetailModal 
        selectedVariable={selectedVariable}
        onClose={() => setSelectedVariable(null)}
        downloadChart={downloadChart}
        filters={filters}
        setFilters={setFilters}
        runAnalysis={runAnalysis}
        handleAddFilter={handleAddFilter}
      />
      
      <JobsHistoryModal 
        isOpen={showHistoryModal}
        onClose={() => setShowHistoryModal(false)}
        history={history}
        onRefresh={() => selectedDataset && loadHistory(selectedDataset)}
        onFetchReport={async (id) => {
            return await EDAService.getReport(id);
        }}
        onSelect={(selectedReport) => {
            setReport(selectedReport);
          // Also update excluded cols state to match the report
          const serverExcluded = Array.isArray(selectedReport.profile_data?.excluded_columns)
            ? selectedReport.profile_data.excluded_columns
            : [];
          setExcludedColsApplied(serverExcluded);
          setExcludedColsDraft(serverExcluded);
        }}
      />
    </div>
  );
};
