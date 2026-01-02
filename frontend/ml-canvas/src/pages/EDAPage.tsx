import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { EDAService } from '../core/api/eda';
import { VariableCard } from '../components/eda/VariableCard';
import { CorrelationHeatmap } from '../components/eda/CorrelationHeatmap';
import { DistributionChart } from '../components/eda/DistributionChart';
import { Loader2, Play, RefreshCw, AlertCircle, BarChart2, Clock, X, Lightbulb, ScatterChart as ScatterIcon, CheckCircle, Info, Map } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, ScatterChart, Scatter, Legend } from 'recharts';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Color palette for PCA clusters
const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57'];

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
  const [activeTab, setActiveTab] = useState('sample');
  const [targetCol, setTargetCol] = useState<string>('');
  const [history, setHistory] = useState<any[]>([]);
  const [selectedVariable, setSelectedVariable] = useState<any>(null);

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
        // Removed auto-switch to target tab
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
            </div>
            <div className="max-h-40 overflow-y-auto pr-2">
                <ul className="space-y-1">
                {profile.alerts.map((alert: any, i: number) => (
                    <li key={i} className="text-sm text-amber-700 dark:text-amber-300 flex items-start">
                    <span className="mr-2">•</span>
                    <span>
                        {alert.column && <span className="font-semibold">{alert.column}: </span>}
                        {alert.message}
                    </span>
                    </li>
                ))}
                </ul>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8">
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
              onClick={() => setActiveTab('insights')}
              className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                activeTab === 'insights'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <Lightbulb className="w-4 h-4" />
              Smart Insights
            </button>
            <button
              onClick={() => setActiveTab('pca')}
              className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                activeTab === 'pca'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <ScatterIcon className="w-4 h-4" />
              Multivariate (PCA)
            </button>
            {profile.geospatial && (
                <button
                onClick={() => setActiveTab('geospatial')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                    activeTab === 'geospatial'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                >
                <Map className="w-4 h-4" />
                Geospatial
                </button>
            )}
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
          </nav>
        </div>

        {/* Tab Content */}
        {activeTab === 'insights' && (
            <div className="mt-4 space-y-6">
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                        <Lightbulb className="w-5 h-5 mr-2 text-yellow-500" />
                        Actionable Recommendations
                    </h3>
                    {profile.recommendations && profile.recommendations.length > 0 ? (
                        <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2">
                            {profile.recommendations.map((rec: any, idx: number) => (
                                <div key={idx} className="flex items-start p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-100 dark:border-gray-800">
                                    <div className={`p-2 rounded-full mr-4 shrink-0 ${
                                        rec.action === 'Drop' ? 'bg-red-100 text-red-600' :
                                        rec.action === 'Impute' ? 'bg-blue-100 text-blue-600' :
                                        rec.action === 'Transform' ? 'bg-purple-100 text-purple-600' :
                                        rec.action === 'Resample' ? 'bg-orange-100 text-orange-600' :
                                        'bg-green-100 text-green-600'
                                    }`}>
                                        {rec.action === 'Drop' ? <X className="w-4 h-4" /> : 
                                            rec.action === 'Keep' ? <CheckCircle className="w-4 h-4" /> :
                                            <RefreshCw className="w-4 h-4" />}
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                                            {rec.action} {rec.column && <span className="text-gray-500">'{rec.column}'</span>}
                                        </h4>
                                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{rec.suggestion}</p>
                                        <p className="text-xs text-gray-400 mt-2 italic">Reason: {rec.reason}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-gray-500 italic">No specific recommendations found. Your data looks clean!</p>
                    )}
                </div>
            </div>
        )}

        {activeTab === 'pca' && (
            <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                    <ScatterIcon className="w-5 h-5 mr-2 text-purple-500" />
                    Multivariate Structure (PCA)
                </h3>

                {profile.pca_data ? (
                    <div className="h-[500px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis 
                                    type="number" 
                                    dataKey="x" 
                                    name="PC1" 
                                    label={{ value: 'Principal Component 1 (PC1)', position: 'bottom', offset: 20, fill: '#6b7280' }} 
                                />
                                <YAxis 
                                    type="number" 
                                    dataKey="y" 
                                    name="PC2" 
                                    label={{ 
                                        value: 'Principal Component 2 (PC2)', 
                                        angle: -90, 
                                        position: 'insideLeft',
                                        style: { textAnchor: 'middle' },
                                        fill: '#6b7280' 
                                    }} 
                                />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Legend verticalAlign="top" height={36}/>
                                {(() => {
                                    // Group data by label for coloring
                                    const groupedData: {[key: string]: any[]} = {};
                                    let hasLabels = false;
                                    
                                    profile.pca_data.forEach((point: any) => {
                                        const label = point.label || 'Data Points';
                                        if (point.label) hasLabels = true;
                                        if (!groupedData[label]) groupedData[label] = [];
                                        groupedData[label].push(point);
                                    });

                                    return Object.keys(groupedData).map((label, index) => (
                                        <Scatter 
                                            key={label} 
                                            name={label} 
                                            data={groupedData[label]} 
                                            fill={hasLabels ? COLORS[index % COLORS.length] : '#8884d8'} 
                                            fillOpacity={0.6} 
                                        />
                                    ));
                                })()}
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded text-gray-400 text-sm text-center p-4">
                        Not enough numeric data for PCA.
                    </div>
                )}

                <div className="mt-6 flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                    <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                    <div className="text-sm text-blue-700 dark:text-blue-300">
                        <p className="font-medium mb-1">About this visualization</p>
                        <p>
                            This chart shows a 2D projection of your data using Principal Component Analysis (PCA). 
                            Points that are close together share similar characteristics across all numeric features.
                        </p>
                        <p className="mt-1 text-xs opacity-80">
                            * For performance, this visualization uses a random sample of up to <strong>5,000 points</strong>.
                        </p>
                    </div>
                </div>
            </div>
        )}

        {activeTab === 'geospatial' && profile.geospatial && (
            <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                        <Map className="w-5 h-5 mr-2 text-blue-500" />
                        Geospatial Analysis
                    </h2>
                    <div className="text-sm text-gray-500">
                        Detected columns: <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lat_col}</span>, <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lon_col}</span>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 h-96 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 relative z-0">
                        {(() => {
                            // Calculate bounds for the map
                            const bounds: [[number, number], [number, number]] = [
                                [profile.geospatial.min_lat, profile.geospatial.min_lon],
                                [profile.geospatial.max_lat, profile.geospatial.max_lon]
                            ];
                            
                            // Helper to get color
                            const uniqueLabels = Array.from(new Set(profile.geospatial.sample_points.map((p: any) => p.label))).filter(Boolean);
                            const isChaotic = uniqueLabels.length > 20;
                            
                            const getColor = (label: string | null) => {
                                if (!label) return '#3b82f6'; // Default blue
                                if (isChaotic) return '#3b82f6'; // Single color for chaotic
                                const index = uniqueLabels.indexOf(label);
                                return COLORS[index % COLORS.length];
                            };

                            return (
                                <MapContainer 
                                    bounds={bounds} 
                                    style={{ height: '100%', width: '100%' }}
                                    scrollWheelZoom={true}
                                >
                                    <TileLayer
                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                    />
                                    {profile.geospatial.sample_points.map((point: any, idx: number) => (
                                        <CircleMarker 
                                            key={idx} 
                                            center={[point.lat, point.lon]} 
                                            radius={5}
                                            pathOptions={{ 
                                                color: getColor(point.label), 
                                                fillColor: getColor(point.label), 
                                                fillOpacity: 0.7,
                                                weight: 1
                                            }}
                                        >
                                            <Popup>
                                                <div className="text-xs">
                                                    <strong>Lat:</strong> {point.lat.toFixed(4)}<br/>
                                                    <strong>Lon:</strong> {point.lon.toFixed(4)}<br/>
                                                    {point.label && <><strong>{profile.target_col}:</strong> {point.label}</>}
                                                </div>
                                            </Popup>
                                        </CircleMarker>
                                    ))}
                                </MapContainer>
                            );
                        })()}
                    </div>

                    <div className="space-y-4">
                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-3">Spatial Bounds</h3>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="block text-xs text-gray-400">Latitude Range</span>
                                    <span className="font-mono">{profile.geospatial.min_lat.toFixed(4)} to {profile.geospatial.max_lat.toFixed(4)}</span>
                                </div>
                                <div>
                                    <span className="block text-xs text-gray-400">Longitude Range</span>
                                    <span className="font-mono">{profile.geospatial.min_lon.toFixed(4)} to {profile.geospatial.max_lon.toFixed(4)}</span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                            <h3 className="text-sm font-medium text-gray-500 mb-3">Centroid</h3>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-red-500"></div>
                                <span className="font-mono text-sm">
                                    {profile.geospatial.centroid_lat.toFixed(4)}, {profile.geospatial.centroid_lon.toFixed(4)}
                                </span>
                            </div>
                        </div>
                        
                        <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                            <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                            <div className="text-sm text-blue-700 dark:text-blue-300">
                                <p>
                                    This scatter plot shows the spatial distribution of your data.
                                    {profile.target_col && " Points are colored by the target variable."}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        )}

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
                                        Strongest predictor: <strong>{Object.keys(profile.target_correlations)[0] || 'None'}</strong> 
                                        {Object.values(profile.target_correlations)[0] !== undefined && (
                                            <> ({(Object.values(profile.target_correlations)[0] as number).toFixed(2)})</>
                                        )}
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
                            history.slice(0, 10).map((item) => (
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
                onClick={() => setSelectedVariable(col)} 
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
        <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Exploratory Data Analysis</h1>
            {report && report.created_at && (
                <p className="text-xs text-gray-500 mt-1">
                    Last analyzed: {new Date(report.created_at).toLocaleString()}
                </p>
            )}
        </div>
        <div className="flex items-center gap-4">
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
      {selectedVariable && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto flex flex-col animate-in fade-in zoom-in duration-200">
            <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-800 z-10">
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  {selectedVariable.name}
                  <span className={`text-sm font-normal px-2 py-0.5 rounded-full ${
                    selectedVariable.dtype === 'Numeric' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                    selectedVariable.dtype === 'Categorical' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                  }`}>
                    {selectedVariable.dtype}
                  </span>
                </h2>
              </div>
              <button 
                onClick={() => setSelectedVariable(null)}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Distribution Chart */}
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4 uppercase tracking-wider">Distribution</h3>
                <DistributionChart profile={selectedVariable} />
              </div>

              {/* Detailed Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* General Stats */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">General Statistics</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                      <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Missing</span>
                      <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.missing_count} ({selectedVariable.missing_percentage.toFixed(2)}%)</span>
                    </div>
                  </div>
                </div>

                {/* Type Specific Stats */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">
                    {selectedVariable.dtype} Statistics
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {selectedVariable.numeric_stats && (
                      <>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Mean</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.mean?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Median</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.median?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Std Dev</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.std?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.min}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.max}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Zeros</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.zeros_count}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Skewness</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.skewness?.toFixed(4)}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Kurtosis</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.kurtosis?.toFixed(4)}</span></div>
                      </>
                    )}
                    {selectedVariable.categorical_stats && (
                      <>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Unique Values</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.unique_count}</span></div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Rare Labels</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.rare_labels_count}</span></div>
                      </>
                    )}
                    {selectedVariable.text_stats && (
                      <>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Avg Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.avg_length?.toFixed(2)}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.min_length}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.max_length}</span></div>
                      </>
                    )}
                    {selectedVariable.date_stats && (
                      <>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.min_date}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.max_date}</span></div>
                         <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Duration (Days)</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.duration_days?.toFixed(1)}</span></div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
