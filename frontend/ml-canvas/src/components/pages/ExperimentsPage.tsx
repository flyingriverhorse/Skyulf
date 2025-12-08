import React, { useState, useEffect } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Filter, Rocket, ChevronDown, ChevronRight } from 'lucide-react';
import { deploymentApi } from '../../core/api/deployment';
import { apiClientV2 } from '../../core/api/client';

export const ExperimentsPage: React.FC = () => {
  const { jobs, fetchJobs } = useJobStore();
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [filterType, setFilterType] = useState<'all' | 'training' | 'tuning'>('all');
  const [datasets, setDatasets] = useState<{id: string, name: string}[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('all');
  
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  
  // Metric visibility toggles
  const [showTrainMetrics, setShowTrainMetrics] = useState(true);
  const [showTestMetrics, setShowTestMetrics] = useState(true);
  const [showValMetrics, setShowValMetrics] = useState(true);

  // Table expansion states
  const [isMetricsExpanded, setIsMetricsExpanded] = useState(true);
  const [isParamsExpanded, setIsParamsExpanded] = useState(true);

  // View state
  const [activeView, setActiveView] = useState<'charts' | 'table'>('charts');

  useEffect(() => {
    fetchJobs();
    fetchDatasets();
  }, [fetchJobs]);

  const fetchDatasets = async () => {
    try {
      const response = await apiClientV2.get('/pipeline/datasets/list');
      setDatasets(response.data);
    } catch (e) {
      console.error("Failed to fetch datasets", e);
    }
  };

  const handleDeploy = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to deploy this model to production?')) return;
    try {
      await deploymentApi.deploy(jobId);
      alert('Model deployed successfully!');
    } catch (err) {
      alert('Failed to deploy model');
      console.error(err);
    }
  };

  const filteredJobs = jobs.filter(job => {
    const typeMatch = filterType === 'all' || job.job_type === filterType;
    const datasetMatch = selectedDatasetId === 'all' || job.dataset_id === selectedDatasetId;
    return typeMatch && datasetMatch;
  });

  const selectedJobs = jobs.filter(job => selectedJobIds.includes(job.job_id));

  const toggleJobSelection = (jobId: string) => {
    setSelectedJobIds(prev => 
      prev.includes(jobId) 
        ? prev.filter(id => id !== jobId)
        : [...prev, jobId]
    );
  };

  // Prepare data for charts
  const metricsData = selectedJobs.map(job => {
    // Use top-level metrics field which is normalized for both types (training & tuning)
    const metrics = job.metrics || job.result?.metrics || {};
    return {
      name: job.job_id.slice(0, 8),
      ...metrics
    };
  });

  // Get all unique metric keys from selected jobs
  const metricKeys = Array.from(new Set(
    selectedJobs.flatMap(job => {
        const m = job.metrics || job.result?.metrics || {};
        // Only include keys that have numeric values to prevent Recharts crashes
        return Object.keys(m).filter(k => {
            const val = m[k];
            return typeof val === 'number' && !isNaN(val);
        });
    })
  )).filter(key => {
      if (key.startsWith('train_') && !showTrainMetrics) return false;
      if (key.startsWith('test_') && !showTestMetrics) return false;
      if (key.startsWith('val_') && !showValMetrics) return false;
      return true;
  });

  const getDuration = (start: string | null, end: string | null) => {
      if (!start || !end) return '-';
      const diff = new Date(end).getTime() - new Date(start).getTime();
      const seconds = Math.floor(diff / 1000);
      if (seconds < 60) return `${seconds}s`;
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ${seconds % 60}s`;
  };

  // Helper to parse keys
  const parseMetricKey = (key: string) => {
      if (key === 'best_score') return { type: 'val', base: 'best_score' };
      if (key.startsWith('train_')) return { type: 'train', base: key.replace('train_', '') };
      if (key.startsWith('test_')) return { type: 'test', base: key.replace('test_', '') };
      if (key.startsWith('val_')) return { type: 'val', base: key.replace('val_', '') };
      return { type: 'other', base: key };
  };

  // Group keys by base metric name
  const metricGroups = new Map<string, string[]>();
  metricKeys.forEach(key => {
      const { base } = parseMetricKey(key);
      if (!metricGroups.has(base)) metricGroups.set(base, []);
      metricGroups.get(base)!.push(key);
  });

  metricGroups.forEach((keys, base) => {
      metricGroups.set(base, keys.sort());
  });

  const availableMetrics = Array.from(metricGroups.keys()).sort();
  const activeMetric = (selectedMetric && availableMetrics.includes(selectedMetric)) 
      ? selectedMetric 
      : availableMetrics[0] || null;

  // const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F'];

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 overflow-hidden">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-100">Experiments & Comparison</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Compare metrics and parameters across multiple runs</p>
        </div>
        <div className="flex gap-2">
           <select 
             className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
             value={selectedDatasetId}
             onChange={(e) => setSelectedDatasetId(e.target.value)}
           >
             <option value="all">All Datasets</option>
             {datasets.map(ds => (
               <option key={ds.id} value={ds.id}>{ds.name}</option>
             ))}
           </select>
           <select 
             className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
             value={filterType}
             onChange={(e) => setFilterType(e.target.value as any)}
           >
             <option value="all">All Experiments</option>
             <option value="tuning">Model Optimization</option>
             <option value="training">Standard Training</option>
           </select>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar List */}
        <div className="w-1/3 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col">
          <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 text-xs font-medium text-gray-500 uppercase">
            Select Runs to Compare ({selectedJobIds.length})
          </div>
          <div className="flex-1 overflow-y-auto">
            {filteredJobs.map(job => (
              <div 
                key={job.job_id}
                onClick={() => toggleJobSelection(job.job_id)}
                className={`p-3 border-b border-gray-100 dark:border-gray-700 cursor-pointer transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 ${
                  selectedJobIds.includes(job.job_id) ? 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-l-blue-500' : 'border-l-4 border-l-transparent'
                }`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-mono text-xs font-semibold text-gray-700 dark:text-gray-300">
                    {job.job_id.slice(0, 8)}
                  </span>
                  <div className="flex items-center gap-2">
                    {job.status === 'completed' && (job.job_type === 'training' || job.job_type === 'tuning') && (
                        <button 
                            onClick={(e) => handleDeploy(e, job.job_id)}
                            className="p-1 hover:bg-blue-100 dark:hover:bg-blue-900 rounded text-blue-600 dark:text-blue-400"
                            title="Deploy to Test"
                        >
                            <Rocket className="w-3 h-3" />
                        </button>
                    )}
                    <span className={`text-[10px] px-1.5 py-0.5 rounded capitalize ${
                        job.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                        job.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                        'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400'
                    }`}>
                        {job.status}
                    </span>
                  </div>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {job.model_type} • {job.dataset_name || 'Unknown Dataset'}
                </div>
                <div className="flex justify-between items-center text-[10px] text-gray-400">
                  <span>{new Date(job.start_time || job.created_at).toLocaleString()}</span>
                  <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-gray-600 dark:text-gray-300">
                    {getDuration(job.start_time, job.end_time)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Comparison Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {selectedJobs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <Filter className="w-12 h-12 mb-4 opacity-20" />
              <p>Select runs from the sidebar to compare them.</p>
            </div>
          ) : (
            <div className="space-y-6">
              
              {/* View Tabs */}
              <div className="flex border-b border-gray-200 dark:border-gray-700">
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'charts'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => setActiveView('charts')}
                  >
                      Visual Comparison
                  </button>
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'table'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => setActiveView('table')}
                  >
                      Detailed Metrics & Params
                  </button>
              </div>

              {/* Charts View */}
              {activeView === 'charts' && (
              <div className="space-y-6 animate-in fade-in duration-300">
                <div className="flex justify-between items-center">
                    <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Metrics Comparison</h3>
                    <div className="flex gap-4 text-sm">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showTrainMetrics} 
                                onChange={e => setShowTrainMetrics(e.target.checked)}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Train</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showTestMetrics} 
                                onChange={e => setShowTestMetrics(e.target.checked)}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Test</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showValMetrics} 
                                onChange={e => setShowValMetrics(e.target.checked)}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Validation</span>
                        </label>
                    </div>
                </div>

                {/* Metric Selector Tabs */}
                {availableMetrics.length > 0 && (
                    <div className="flex flex-wrap gap-2 border-b border-gray-200 dark:border-gray-700 pb-2">
                        {availableMetrics.map(metric => (
                            <button
                                key={metric}
                                onClick={() => setSelectedMetric(metric)}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                                    activeMetric === metric
                                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                                }`}
                            >
                                {metric.toUpperCase()}
                            </button>
                        ))}
                    </div>
                )}

                {activeMetric && metricGroups.has(activeMetric) && (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                        <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={metricsData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                            <XAxis dataKey="name" stroke="#6B7280" fontSize={12} />
                            <YAxis stroke="#6B7280" fontSize={12} />
                            <Tooltip 
                                shared={false}
                                cursor={{ fill: 'rgba(107, 114, 128, 0.1)' }}
                                contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#F3F4F6' }}
                                itemStyle={{ color: '#F3F4F6' }}
                            />
                            <Legend />
                            {metricGroups.get(activeMetric)!.map((key) => {
                                const { type } = parseMetricKey(key);
                                // Colors: Train=Blue, Test=Green, Val=Orange, Other=Purple
                                let color = '#8884d8';
                                if (type === 'train') color = '#3b82f6'; // blue-500
                                if (type === 'test') color = '#22c55e'; // green-500
                                if (type === 'val') color = '#f97316'; // orange-500
                                if (type === 'other') color = '#a855f7'; // purple-500
                                
                                return <Bar key={key} dataKey={key} fill={color} name={`${type} (${activeMetric})`} />;
                            })}
                            </BarChart>
                        </ResponsiveContainer>
                        </div>
                    </div>
                )}
              </div>
              )}

              {/* Table View */}
              {activeView === 'table' && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden animate-in fade-in duration-300">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Detailed Comparison</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-900/50 dark:text-gray-400">
                      <tr>
                        <th className="px-4 py-2">Parameter / Metric</th>
                        {selectedJobs.map(job => (
                          <th key={job.job_id} className="px-4 py-2 font-mono">
                            {job.job_id.slice(0, 8)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {/* Model Type */}
                      <tr className="bg-white dark:bg-gray-800">
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100">Model Type</td>
                        {selectedJobs.map(job => (
                          <td key={job.job_id} className="px-4 py-2 text-gray-500 dark:text-gray-400">
                            {job.model_type}
                          </td>
                        ))}
                      </tr>
                      {/* Metrics Section in Table */}
                      <tr 
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => setIsMetricsExpanded(!isMetricsExpanded)}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isMetricsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Key Metrics
                        </td>
                      </tr>
                      {isMetricsExpanded && metricKeys.map(metricKey => (
                        <tr key={metricKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">{metricKey}</td>
                          {selectedJobs.map(job => {
                             const m = job.metrics || job.result?.metrics || {};
                             const val = m[metricKey];
                             return (
                                <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                                  {typeof val === 'number' ? val.toFixed(4) : '-'}
                                </td>
                             );
                          })}
                        </tr>
                      ))}
                      {/* Hyperparameters */}
                      <tr 
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => setIsParamsExpanded(!isParamsExpanded)}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isParamsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Hyperparameters
                        </td>
                      </tr>
                      {/* Extract all unique hyperparameter keys */}
                      {isParamsExpanded && Array.from(new Set(selectedJobs.flatMap(job => Object.keys(job.hyperparameters || {})))).map(paramKey => (
                        <tr key={paramKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">{paramKey}</td>
                          {selectedJobs.map(job => (
                            <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                              {String(job.hyperparameters?.[paramKey] ?? '-')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
