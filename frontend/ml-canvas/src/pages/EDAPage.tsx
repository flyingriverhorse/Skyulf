import React, { useState, useEffect } from 'react';
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
import { Loader2, RefreshCw, AlertCircle, BarChart2, List, Play } from 'lucide-react';
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
  const [excludedCols, setExcludedCols] = useState<string[]>([]);
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

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      setReport(null); // Clear previous report to show loader
      setExcludedCols([]); // Reset excluded columns
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
      setExcludedCols([]);
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
    const actualExcluded = Array.isArray(overrideExcluded) ? overrideExcluded : excludedCols;
    const actualFilters = Array.isArray(overrideFilters) ? overrideFilters : filters;

    if (!selectedDataset) return;
    setAnalyzing(true);
    try {
      await EDAService.analyze(selectedDataset, targetCol || undefined, actualExcluded, actualFilters);
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
    const message = exclude 
        ? `Are you sure you want to exclude '${colName}' from the analysis? This will trigger a new analysis.`
        : `Include '${colName}' back in the analysis? This will trigger a new analysis.`;
        
    if (confirm(message)) {
        let newExcluded = [...excludedCols];
        if (exclude) {
            newExcluded.push(colName);
        } else {
            newExcluded = newExcluded.filter(c => c !== colName);
        }
        setExcludedCols(newExcluded);
        runAnalysis(newExcluded);
    }
  };

  // Sync target col and excluded cols from report if available
  useEffect(() => {
    if (report && report.profile_data) {
        if (report.profile_data.target_col) {
            setTargetCol(report.profile_data.target_col);
        }
        if (report.profile_data.excluded_columns) {
            setExcludedCols(report.profile_data.excluded_columns);
        } else {
            setExcludedCols([]);
        }
    }
  }, [report]);

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
            <div className="w-64">
                <input
                type="text"
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="Target Column (Optional)"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                />
            </div>
            <button
                onClick={() => runAnalysis()}
                disabled={analyzing}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
                {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                Run Analysis
            </button>
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

    const profile = report.profile_data;
    if (!profile) return <div>No profile data</div>;

    return (
      <div className="flex h-[calc(100vh-124px)] border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
        <EDASidebar 
            activeTab={activeTab} 
            setActiveTab={setActiveTab} 
            profile={profile} 
            filters={filters}
            columns={profile.columns ? Object.keys(profile.columns) : []}
            excludedCols={excludedCols}
            onAddFilter={handleAddFilter}
            onRemoveFilter={handleRemoveFilter}
            onClearFilters={() => { setFilters([]); runAnalysis(undefined, []); }}
            onToggleExclude={handleToggleExclude}
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
                    columns={report?.profile_data?.columns ? Object.keys(report.profile_data.columns) : []}
                    initialFilters={filters}
                />
            )}

            {activeTab === 'sample' && profile.sample_data && (
                <SampleDataTab 
                    profile={profile}
                    excludedCols={excludedCols}
                    handleToggleExclude={handleToggleExclude}
                />
            )}
        </div>
      </div>
    );
  };

  return (
    <div id="eda-report-container" className="p-6 max-w-7xl mx-auto">
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
                {report && report.profile_data && Object.keys(report.profile_data.columns).map(col => (
                    <option key={col} value={col}>{col}</option>
                ))}
             </select>
          </div>

          <button 
            onClick={() => selectedDataset && runAnalysis()}
            disabled={!selectedDataset || analyzing}
            className="p-2 text-gray-500 hover:text-blue-600 rounded-full hover:bg-gray-100"
            title="Refresh Analysis"
          >
            <RefreshCw className={`w-5 h-5 ${analyzing ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

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
            if (selectedReport.profile_data?.excluded_columns) {
                setExcludedCols(selectedReport.profile_data.excluded_columns);
            } else {
                setExcludedCols([]);
            }
        }}
      />
    </div>
  );
};
