import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { EDAService } from '../core/api/eda';
import { VariableCard } from '../components/eda/VariableCard';
import { CorrelationHeatmap } from '../components/eda/CorrelationHeatmap';
import { Loader2, Play, RefreshCw, AlertCircle, BarChart2, Clock } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';

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
  const [activeTab, setActiveTab] = useState('variables');
  const [showAllAlerts, setShowAllAlerts] = useState(false);
  const [targetCol, setTargetCol] = useState<string>('');
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      setReport(null); // Clear previous report to show loader
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } else {
      setReport(null);
      setHistory([]);
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

  const runAnalysis = async () => {
    if (!selectedDataset) return;
    setAnalyzing(true);
    try {
      await EDAService.analyze(selectedDataset, targetCol || undefined);
      // Reload immediately to get the PENDING state
      loadReport(selectedDataset);
      loadHistory(selectedDataset);
    } catch (err) {
      setError("Failed to start analysis");
    } finally {
      setAnalyzing(false);
    }
  };

  // Sync target col from report if available
  useEffect(() => {
    if (report && report.profile_data && report.profile_data.target_col) {
        setTargetCol(report.profile_data.target_col);
        setActiveTab('target'); // Switch to target tab automatically
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
                onClick={runAnalysis}
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
            onClick={runAnalysis}
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
      <div className="space-y-6">
        {/* Analysis Controls */}
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
                <div className="flex flex-col">
                    <label className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Target Column</label>
                    <select 
                        value={targetCol || profile.target_col || ''}
                        onChange={(e) => setTargetCol(e.target.value)}
                        className="block w-48 px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    >
                        <option value="">None</option>
                        {profile.columns && Object.keys(profile.columns).map((col: string) => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>
                <button
                    onClick={runAnalysis}
                    disabled={analyzing}
                    className="mt-5 flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
                >
                    {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-2" />}
                    Update
                </button>
            </div>
             <div className="text-sm text-gray-500">
                {report.created_at && <span>Analyzed: {new Date(report.created_at).toLocaleString()}</span>}
             </div>
        </div>

        {/* Overview Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Rows</div>
            <div className="text-2xl font-bold">{profile.row_count.toLocaleString()}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Columns</div>
            <div className="text-2xl font-bold">{profile.column_count}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Missing Cells</div>
            <div className="text-2xl font-bold">{profile.missing_cells_percentage.toFixed(1)}%</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-sm text-gray-500">Duplicates</div>
            <div className="text-2xl font-bold">{profile.duplicate_rows}</div>
          </div>
        </div>

        {/* Alerts */}
        {profile.alerts && profile.alerts.length > 0 && (
          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium text-amber-800 dark:text-amber-200 flex items-center">
                <AlertCircle className="w-4 h-4 mr-2" />
                Data Quality Alerts ({profile.alerts.length})
              </h3>
              {profile.alerts.length > 5 && (
                <button
                  onClick={() => setShowAllAlerts(!showAllAlerts)}
                  className="text-xs text-amber-700 hover:text-amber-900 underline"
                >
                  {showAllAlerts ? "Show Less" : "Show All"}
                </button>
              )}
            </div>
            <ul className="space-y-1">
              {(showAllAlerts ? profile.alerts : profile.alerts.slice(0, 5)).map((alert: any, i: number) => (
                <li key={i} className="text-sm text-amber-700 dark:text-amber-300 flex items-start">
                  <span className="mr-2">•</span>
                  <span>
                    {alert.column && <span className="font-semibold">{alert.column}: </span>}
                    {alert.message}
                  </span>
                </li>
              ))}
            </ul>
            {!showAllAlerts && profile.alerts.length > 5 && (
              <div className="text-xs text-amber-600 mt-2 italic">
                + {profile.alerts.length - 5} more alerts...
              </div>
            )}
          </div>
        )}

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('variables')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'variables'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Variables
            </button>
            <button
              onClick={() => setActiveTab('correlations')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'correlations'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Correlations
            </button>
            {profile.target_col && (
                <button
                onClick={() => setActiveTab('target')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'target'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                Target Analysis
                </button>
            )}
            <button
              onClick={() => setActiveTab('sample')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'sample'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Sample Data
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'target' && profile.target_col && profile.target_correlations && (
            <div className="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Analysis Content */}
                <div className="lg:col-span-3 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                            <BarChart2 className="w-5 h-5 mr-2 text-blue-500" />
                            Target Analysis: <span className="ml-2 text-blue-600">{profile.target_col}</span>
                        </h2>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2">
                            <h3 className="text-sm font-medium text-gray-500 mb-4">Top Associated Features</h3>
                            <div className="h-64 w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart 
                                        data={Object.entries(profile.target_correlations)
                                            .slice(0, 10)
                                            .map(([k, v]) => ({ name: k, value: v as number }))
                                        } 
                                        layout="vertical"
                                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                    >
                                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                                        <XAxis type="number" domain={[0, 1]} />
                                        <YAxis type="category" dataKey="name" width={120} tick={{fontSize: 12}} />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}
                                            itemStyle={{ color: '#fff' }}
                                            formatter={(value: number) => [value.toFixed(3), 'Association']}
                                        />
                                        <ReferenceLine x={0} stroke="#9ca3af" />
                                        <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-2">Key Insights</h3>
                            <ul className="space-y-2 text-sm">
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>Target is <strong>{profile.columns[profile.target_col]?.dtype}</strong></span>
                                </li>
                                <li className="flex items-start">
                                    <span className="mr-2 text-blue-500">•</span>
                                    <span>
                                        Strongest predictor: <strong>{Object.keys(profile.target_correlations)[0]}</strong> 
                                        ({(Object.values(profile.target_correlations)[0] as number).toFixed(2)})
                                    </span>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* History Sidebar */}
                <div className="lg:col-span-1 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm h-fit">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center justify-between">
                        <div className="flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            Analysis History
                        </div>
                        {loading && <Loader2 className="w-3 h-3 animate-spin text-blue-500" />}
                    </h3>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                        {history.length === 0 ? (
                            <p className="text-xs text-gray-500 italic">No previous analyses found.</p>
                        ) : (
                            history.map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => !loading && loadSpecificReport(item.id)}
                                    disabled={loading}
                                    className={`w-full text-left p-3 rounded-md text-sm border transition-colors ${
                                        report && report.id === item.id
                                            ? 'bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300'
                                            : 'bg-gray-50 border-gray-100 text-gray-600 hover:bg-gray-100 dark:bg-gray-900 dark:border-gray-800 dark:text-gray-400 dark:hover:bg-gray-800'
                                    } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                                >
                                    <div className="font-medium truncate">
                                        {item.target_col ? `Target: ${item.target_col}` : 'General Analysis'}
                                    </div>
                                    <div className="text-xs opacity-70 mt-1">
                                        {new Date(item.created_at).toLocaleString()}
                                    </div>
                                    <div className={`text-xs mt-1 inline-block px-1.5 py-0.5 rounded ${
                                        item.status === 'COMPLETED' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' :
                                        item.status === 'FAILED' ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                                        'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                                    }`}>
                                        {item.status}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>
            </div>
        )}

        {/* Tab Content */}
        {activeTab === 'variables' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {Object.values(profile.columns).map((col: any) => (
              <VariableCard 
                key={col.name} 
                profile={col} 
                onClick={() => console.log("Clicked", col.name)} 
              />
            ))}
          </div>
        )}

        {activeTab === 'correlations' && profile.correlations && (
          <div className="mt-4">
            <CorrelationHeatmap data={profile.correlations} />
          </div>
        )}

        {activeTab === 'sample' && profile.sample_data && (
          <div className="mt-4 overflow-x-auto bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  {Object.keys(profile.sample_data[0] || {}).map((col) => (
                    <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {profile.sample_data.map((row: any, i: number) => (
                  <tr key={i}>
                    {Object.values(row).map((val: any, j: number) => (
                      <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {val !== null ? String(val) : <span className="italic text-gray-400">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory Data Analysis</h1>
        <div className="flex items-center gap-4">
          <select
            value={selectedDataset || ''}
            onChange={(e) => setSelectedDataset(Number(e.target.value))}
            className="block w-64 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border"
          >
            <option value="" disabled>Select a dataset</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>{ds.name}</option>
            ))}
          </select>
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
    </div>
  );
};
