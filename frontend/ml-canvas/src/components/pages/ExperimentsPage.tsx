import React, { useState, useEffect } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, LineChart, Line, ReferenceLine
} from 'recharts';
import { Filter, Rocket, ChevronDown, ChevronRight, Activity, RefreshCw } from 'lucide-react';
import { deploymentApi } from '../../core/api/deployment';
import { apiClient } from '../../core/api/client';

export const ExperimentsPage: React.FC = () => {
  const { jobs, fetchJobs, hasMore, loadMoreJobs, isLoading } = useJobStore();
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
  const [activeView, setActiveView] = useState<'charts' | 'table' | 'evaluation'>('charts');
  const [evaluationData, setEvaluationData] = useState<any>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalJobId, setEvalJobId] = useState<string | null>(null);
  const [selectedRocClass, setSelectedRocClass] = useState<string | null>(null);

  useEffect(() => {
    fetchJobs();
    fetchDatasets();
  }, [fetchJobs]);

  useEffect(() => {
    if (evaluationData?.splits?.train?.y_proba?.classes?.length > 0) {
        setSelectedRocClass(evaluationData.splits.train.y_proba.classes[0]);
    }
  }, [evaluationData]);

  const fetchDatasets = async () => {
    try {
      const response = await apiClient.get('/pipeline/datasets/list');
      setDatasets(response.data);
    } catch (e) {
      console.error("Failed to fetch datasets", e);
    }
  };

  const handleDeploy = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to deploy this model to production?')) return;
    try {
      await deploymentApi.deployModel(jobId);
      alert('Model deployed successfully!');
    } catch (err) {
      alert('Failed to deploy model');
      console.error(err);
    }
  };

  const fetchEvaluationData = async (jobId: string) => {
      setIsEvalLoading(true);
      setEvaluationData(null);
      setEvalError(null);
      setEvalJobId(jobId);
      try {
          const res = await apiClient.get(`/pipeline/jobs/${jobId}/evaluation`);
          setEvaluationData(res.data);
      } catch (err: any) {
          console.error("Failed to fetch evaluation data", err);
          setEvalError(err.response?.data?.detail || "Failed to fetch evaluation data");
      } finally {
          setIsEvalLoading(false);
      }
  };

  const calculateConfusionMatrix = (y_true: any[], y_pred: any[]) => {
      const classes = Array.from(new Set([...y_true, ...y_pred])).sort();
      const matrix = classes.map(trueClass => {
          return classes.map(predClass => {
              return y_true.reduce((count, t, i) => {
                  const p = y_pred[i];
                  return (t === trueClass && p === predClass) ? count + 1 : count;
              }, 0);
          });
      });
      return { classes, matrix };
  };

  const calculateROC = (y_true: any[], y_proba: { classes: any[], values: number[][] }, targetClass: any) => {
      if (!y_proba || !y_proba.values) return null;
      
      // Find index of target class
      const classIndex = y_proba.classes.indexOf(targetClass);
      if (classIndex === -1) return null;
      
      const scores = y_proba.values.map(v => v[classIndex]);
      
      const data = scores.map((score, i) => ({
          score,
          actual: y_true[i] === targetClass ? 1 : 0
      }));
      
      // Sort by score descending
      data.sort((a, b) => b.score - a.score);
      
      const rocPoints = [];
      let tp = 0;
      let fp = 0;
      const totalPos = data.filter(d => d.actual === 1).length;
      const totalNeg = data.length - totalPos;
      
      if (totalPos === 0 || totalNeg === 0) return null; // Cannot calculate ROC if only one class present in split

      rocPoints.push({ fpr: 0, tpr: 0 });
      
      for (let i = 0; i < data.length; i++) {
          if (data[i].actual === 1) tp++;
          else fp++;
          
          rocPoints.push({
              fpr: fp / totalNeg,
              tpr: tp / totalPos
          });
      }
      
      return rocPoints;
  };

  // Effect to fetch evaluation data when view changes or selection changes
  useEffect(() => {
      if (activeView === 'evaluation') {
          // If we have a specific eval job selected, use it
          // Otherwise default to the first selected job
          if (!evalJobId && selectedJobIds.length > 0) {
              fetchEvaluationData(selectedJobIds[0]);
          } else if (evalJobId && !selectedJobIds.includes(evalJobId) && selectedJobIds.length > 0) {
              // If current eval job is deselected, switch to another
              fetchEvaluationData(selectedJobIds[0]);
          } else if (selectedJobIds.length === 0) {
              setEvaluationData(null);
              setEvalJobId(null);
          }
      }
  }, [activeView, selectedJobIds, evalJobId]);

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
                  {job.job_type === 'tuning' && job.config?.tuning?.strategy && (
                      <span className="ml-1 text-gray-400">
                          ({job.config.tuning.strategy})
                      </span>
                  )}
                </div>
                <div className="flex justify-between items-center text-[10px] text-gray-400">
                  <span>{new Date(job.start_time || job.created_at).toLocaleString()}</span>
                  <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-gray-600 dark:text-gray-300">
                    {getDuration(job.start_time, job.end_time)}
                  </span>
                </div>
              </div>
            ))}
            
            {hasMore && (
                <div className="p-2 flex justify-center border-t border-gray-100 dark:border-gray-700">
                    <button 
                        onClick={() => loadMoreJobs()}
                        disabled={isLoading}
                        className="text-xs text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50 flex items-center gap-1"
                    >
                        {isLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <ChevronDown className="w-3 h-3" />}
                        Load More
                    </button>
                </div>
            )}
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
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'evaluation'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => setActiveView('evaluation')}
                  >
                      Model Evaluation
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

              {/* Evaluation View */}
              {activeView === 'evaluation' && (
                <div className="space-y-6 animate-in fade-in duration-300">
                    {/* Job Selector if multiple */}
                    {selectedJobIds.length > 1 && (
                        <div className="flex gap-2 overflow-x-auto pb-2">
                            {selectedJobIds.map(id => (
                                <button
                                    key={id}
                                    onClick={() => fetchEvaluationData(id)}
                                    className={`px-3 py-1 text-xs font-mono rounded border ${
                                        evalJobId === id 
                                            ? 'bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300'
                                            : 'bg-white border-gray-200 text-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400'
                                    }`}
                                >
                                    {id.slice(0, 8)}
                                </button>
                            ))}
                        </div>
                    )}

                    {isEvalLoading ? (
                        <div className="h-64 flex items-center justify-center text-gray-400">
                            <Activity className="w-6 h-6 animate-spin mr-2" />
                            Loading evaluation data...
                        </div>
                    ) : evalError ? (
                        <div className="h-64 flex flex-col items-center justify-center text-red-500 p-4 text-center">
                            <p className="font-medium">Error loading evaluation data</p>
                            <p className="text-sm mt-2 opacity-80">{evalError}</p>
                        </div>
                    ) : !evaluationData ? (
                        <div className="h-64 flex items-center justify-center text-gray-400 italic text-center">
                            <p>Select a completed job to view evaluation details.</p>
                            <p className="text-xs mt-2 opacity-70">(Note: Only jobs run after this update have evaluation artifacts)</p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {/* Controls for Evaluation View */}
                            <div className="flex justify-between items-center bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
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
                                
                                {evaluationData.problem_type === 'classification' && evaluationData.splits?.train?.y_proba && (
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm text-gray-500 dark:text-gray-400">Target Class (ROC):</span>
                                        <select 
                                            className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-1.5"
                                            value={selectedRocClass || ''}
                                            onChange={(e) => setSelectedRocClass(e.target.value)}
                                        >
                                            {evaluationData.splits.train.y_proba.classes.map((c: any) => (
                                                <option key={c} value={c}>{c}</option>
                                            ))}
                                        </select>
                                    </div>
                                )}
                            </div>

                            <div className="flex flex-col gap-6">
                            {/* Render charts for each split */}
                            {Object.entries(evaluationData.splits || {})
                                .filter(([splitName]) => {
                                    if (splitName === 'train' && !showTrainMetrics) return false;
                                    if (splitName === 'test' && !showTestMetrics) return false;
                                    if (splitName === 'validation' && !showValMetrics) return false;
                                    return true;
                                })
                                .map(([splitName, splitData]: [string, any]) => {
                                const data = splitData.y_true.map((y: any, i: number) => ({
                                    x: y,
                                    y: splitData.y_pred[i]
                                }));
                                
                                return (
                                    <div key={splitName} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 capitalize">{splitName} Set</h4>
                                        
                                        {evaluationData.problem_type === 'regression' ? (
                                            <div className="grid grid-cols-1 gap-8">
                                                {/* Scatter Plot: Actual vs Predicted */}
                                                <div className="h-[300px]">
                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center">Actual vs Predicted</h5>
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                                            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                                                            <XAxis 
                                                                type="number" 
                                                                dataKey="x" 
                                                                name="Actual" 
                                                                unit="" 
                                                                label={{ value: 'Actual Values', position: 'bottom', offset: 0, fontSize: 12 }} 
                                                                tick={{ fontSize: 11 }}
                                                            />
                                                            <YAxis 
                                                                type="number" 
                                                                dataKey="y" 
                                                                name="Predicted" 
                                                                unit="" 
                                                                label={{ value: 'Predicted Values', angle: -90, position: 'left', fontSize: 12 }} 
                                                                tick={{ fontSize: 11 }}
                                                            />
                                                            <Tooltip 
                                                                cursor={{ strokeDasharray: '3 3' }}
                                                                content={({ active, payload }) => {
                                                                    if (active && payload && payload.length) {
                                                                        const data = payload[0].payload;
                                                                        return (
                                                                            <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                                                                                <p className="font-medium">Actual: {data.x.toFixed(4)}</p>
                                                                                <p className="font-medium">Predicted: {data.y.toFixed(4)}</p>
                                                                                <p className="text-gray-500">Error: {(data.y - data.x).toFixed(4)}</p>
                                                                            </div>
                                                                        );
                                                                    }
                                                                    return null;
                                                                }}
                                                            />
                                                            {/* Reference Line for Perfect Prediction (y=x) */}
                                                            <Line 
                                                                dataKey="x" 
                                                                stroke="#ccc" 
                                                                strokeDasharray="3 3" 
                                                                dot={false} 
                                                                activeDot={false} 
                                                                legendType="none"
                                                                isAnimationActive={false}
                                                            />
                                                            <Scatter name="Predictions" data={data} fill="#8884d8" fillOpacity={0.6} />
                                                        </ScatterChart>
                                                    </ResponsiveContainer>
                                                </div>

                                                {/* Residual Plot: Residuals vs Predicted */}
                                                <div className="h-[300px]">
                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center">Residuals vs Predicted</h5>
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                                            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                                                            <XAxis 
                                                                type="number" 
                                                                dataKey="y" 
                                                                name="Predicted" 
                                                                unit="" 
                                                                label={{ value: 'Predicted Values', position: 'bottom', offset: 0, fontSize: 12 }} 
                                                                tick={{ fontSize: 11 }}
                                                            />
                                                            <YAxis 
                                                                type="number" 
                                                                dataKey="residual" 
                                                                name="Residual" 
                                                                unit="" 
                                                                label={{ value: 'Residuals (Actual - Predicted)', angle: -90, position: 'left', fontSize: 12 }} 
                                                                tick={{ fontSize: 11 }}
                                                            />
                                                            <Tooltip 
                                                                cursor={{ strokeDasharray: '3 3' }}
                                                                content={({ active, payload }) => {
                                                                    if (active && payload && payload.length) {
                                                                        const data = payload[0].payload;
                                                                        return (
                                                                            <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                                                                                <p className="font-medium">Predicted: {data.y.toFixed(4)}</p>
                                                                                <p className="font-medium">Residual: {data.residual.toFixed(4)}</p>
                                                                            </div>
                                                                        );
                                                                    }
                                                                    return null;
                                                                }}
                                                            />
                                                            {/* Zero Line */}
                                                            <ReferenceLine y={0} stroke="#ccc" strokeDasharray="3 3" />
                                                            <Scatter 
                                                                name="Residuals" 
                                                                data={data.map((d: any) => ({ ...d, residual: d.x - d.y }))} 
                                                                fill="#82ca9d" 
                                                                fillOpacity={0.6} 
                                                            />
                                                        </ScatterChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                                {/* Confusion Matrix */}
                                                <div className="h-[300px] flex flex-col items-center justify-center">
                                                    {(() => {
                                                        const { classes, matrix } = calculateConfusionMatrix(splitData.y_true, splitData.y_pred);
                                                        return (
                                                            <div className="flex flex-col items-center">
                                                                <div className="flex">
                                                                    <div className="w-8"></div> {/* Y-axis label spacer */}
                                                                    <div className="flex ml-2"> {/* Added margin-left to align with matrix */}
                                                                        {classes.map(c => (
                                                                            <div key={c} className="w-16 text-center text-xs font-medium text-gray-500 dark:text-gray-400 pb-2">
                                                                                Pred {c}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                </div>
                                                                <div className="flex">
                                                                    <div className="flex flex-col justify-center mr-2">
                                                                        {classes.map(c => (
                                                                            <div key={c} className="h-16 flex items-center justify-end text-xs font-medium text-gray-500 dark:text-gray-400 pr-2">
                                                                                True {c}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                    <div className="border border-gray-200 dark:border-gray-700">
                                                                        {matrix.map((row, i) => (
                                                                            <div key={i} className="flex">
                                                                                {row.map((count, j) => {
                                                                                    // Calculate intensity
                                                                                    const max = Math.max(...matrix.flat());
                                                                                    const intensity = max > 0 ? count / max : 0;
                                                                                    return (
                                                                                        <div 
                                                                                            key={j} 
                                                                                            className="w-16 h-16 flex items-center justify-center text-sm font-mono border border-gray-100 dark:border-gray-800"
                                                                                            style={{
                                                                                                backgroundColor: `rgba(59, 130, 246, ${intensity * 0.8 + 0.1})`, // Blue base
                                                                                                color: intensity > 0.5 ? 'white' : 'inherit'
                                                                                            }}
                                                                                            title={`True: ${classes[i]}, Pred: ${classes[j]}, Count: ${count}`}
                                                                                        >
                                                                                            {count}
                                                                                        </div>
                                                                                    );
                                                                                })}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                </div>
                                                                <div className="mt-4 text-xs text-gray-400">
                                                                    Confusion Matrix
                                                                </div>
                                                            </div>
                                                        );
                                                    })()}
                                                </div>

                                                {/* ROC Curve (if available) */}
                                                {splitData.y_proba && (
                                                    <div className="h-[300px] w-full">
                                                        <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center">ROC Curve</h5>
                                                        {(() => {
                                                            const rocData = calculateROC(splitData.y_true, splitData.y_proba, selectedRocClass);
                                                            if (!rocData) return <div className="text-center text-xs text-gray-400">ROC not available (multiclass or missing proba)</div>;
                                                            
                                                            return (
                                                                <ResponsiveContainer width="100%" height="100%">
                                                                    <LineChart data={rocData} margin={{ top: 5, right: 20, bottom: 20, left: 20 }}>
                                                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                                                        <XAxis type="number" dataKey="fpr" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'bottom', offset: 0, fontSize: 12 }} />
                                                                        <YAxis type="number" dataKey="tpr" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'left', fontSize: 12 }} />
                                                                        <Tooltip />
                                                                        <Line type="monotone" dataKey="tpr" stroke="#8884d8" dot={false} strokeWidth={2} />
                                                                        {/* Diagonal reference line */}
                                                                        <Line dataKey="fpr" stroke="#ccc" strokeDasharray="3 3" dot={false} />
                                                                    </LineChart>
                                                                </ResponsiveContainer>
                                                            );
                                                        })()}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                        </div>
                    )}
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
