import React, { useState, useEffect } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, Line, ReferenceLine, ComposedChart, Area
} from 'recharts';
import { clickableProps } from '../../core/utils/a11y';
import { Filter, Rocket, ChevronDown, ChevronRight, ChevronLeft, RefreshCw, Download, Loader2, Check, Trophy, GitBranch } from 'lucide-react';
import { LoadingState, ErrorState, useConfirm } from '../shared';
import { toast } from '../../core/toast';
import { toPng } from 'html-to-image';
import { deploymentApi } from '../../core/api/deployment';
import { apiClient } from '../../core/api/client';
import { formatMetricName, getMetricDescription, getHyperparamDescription, getTrainingConfigDescription } from '../../core/utils/format';
import { InfoTooltip } from '../ui/InfoTooltip';
import { PipelineDiffView } from './experiments/PipelineDiffView';

/** Extract the resolved scoring metric from a job's result (top-level or nested in metrics). */
function getJobScoringMetric(job: { result?: Record<string, unknown> | null }): string | undefined {
  const r = job.result;
  if (r?.scoring_metric) return r.scoring_metric as string;
  const m = r?.metrics as Record<string, unknown> | undefined;
  if (m?.scoring_metric) return m.scoring_metric as string;
  return undefined;
}

/** Extract a short 8-char run ID from a pipeline_id, stripping the "preview_" prefix
 *  and any "__branch_N" suffix so all experiments from the same batch share the same ID. */
function shortRunId(job: { pipeline_id: string; parent_pipeline_id?: string | null }): string {
  const raw = job.parent_pipeline_id || job.pipeline_id;
  // Strip "preview_" prefix and any "__branch_*" suffix
  const clean = raw.replace(/^preview_/, '').replace(/__branch_.*$/, '');
  return clean.slice(0, 8);
}

interface EvaluationSplit {
  y_true: (string | number)[];
  y_pred: (string | number)[];
  y_proba?: {
    classes: (string | number)[];
        labels?: (string | number)[];
    values: number[][];
  };
  metrics?: Record<string, number>;
}

interface EvaluationData {
  problem_type: 'classification' | 'regression';
  splits: Record<string, EvaluationSplit>;
}

export const ExperimentsPage: React.FC = () => {
  const { jobs, fetchJobs, hasMore, loadMoreJobs, isLoading, promoteJob, unpromoteJob } = useJobStore();
  const confirm = useConfirm();
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [filterType, setFilterType] = useState<'all' | 'basic_training' | 'advanced_tuning'>('all');
  const [datasets, setDatasets] = useState<{id: string, name: string}[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('all');
  
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  
  // Metric visibility toggles
  const [showTrainMetrics, setShowTrainMetrics] = useState(true);
  const [showTestMetrics, setShowTestMetrics] = useState(true);
  const [showValMetrics, setShowValMetrics] = useState(true);
  const [showCvMetrics, setShowCvMetrics] = useState(true);

  // Table expansion states
  const [isMetricsExpanded, setIsMetricsExpanded] = useState(true);
  const [isParamsExpanded, setIsParamsExpanded] = useState(true);
  const [isTuningExpanded, setIsTuningExpanded] = useState(true);
  const [isPipelineExpanded, setIsPipelineExpanded] = useState(true);

  // View state
  const [activeView, setActiveView] = useState<'charts' | 'table' | 'evaluation' | 'importance' | 'diff'>('charts');
  const [evaluationData, setEvaluationData] = useState<EvaluationData | null>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalJobId, setEvalJobId] = useState<string | null>(null);
  const [downloadingChart, setDownloadingChart] = useState<string | null>(null);
  const [doneChart, setDoneChart] = useState<string | null>(null);
  const [selectedRocClass, setSelectedRocClass] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [cmView, setCmView] = useState<'overall' | 'per-class'>('overall');

  useEffect(() => {
    fetchJobs();
    void fetchDatasets();
  }, [fetchJobs]);

  useEffect(() => {
    if (evaluationData?.splits.train?.y_proba?.classes && evaluationData.splits.train.y_proba.classes.length > 0) {
                const proba = evaluationData.splits.train.y_proba;
                const first = proba.labels?.[0] ?? proba.classes[0];
                setSelectedRocClass(String(first));
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
    const ok = await confirm({
      title: 'Deploy to production?',
      message: 'Are you sure you want to deploy this model to production?',
      confirmLabel: 'Deploy',
    });
    if (!ok) return;
    try {
      await deploymentApi.deployModel(jobId);
      toast.success('Model deployed');
    } catch {
      toast.error('Failed to deploy model');
    }
  };

  const handlePromote = async (e: React.MouseEvent, job: typeof jobs[0]) => {
    e.stopPropagation();
    try {
      if (job.promoted_at) {
        await unpromoteJob(job.job_id);
      } else {
        await promoteJob(job.job_id);
      }
    } catch {
      toast.error('Failed to update promotion status');
    }
  };

  const handleDownload = async (elementId: string, fileName: string) => {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    setDownloadingChart(elementId);
    const isDarkMode = document.documentElement.classList.contains('dark');
    const backgroundColor = isDarkMode ? '#1f2937' : '#ffffff';

    try {
        const dataUrl = await toPng(element, {
            backgroundColor,
            pixelRatio: 2,
            filter: (node) => !(node instanceof HTMLElement && node.dataset.exportIgnore === 'true'),
        });
        const link = document.createElement('a');
        link.download = `${fileName}.png`;
        link.href = dataUrl;
        link.click();
    } catch (e) {
        toast.error('Image download failed', String(e));
    } finally {
        setDownloadingChart(null);
        setDoneChart(elementId);
        setTimeout(() => setDoneChart(null), 1200);
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
      } catch (err: unknown) {
          console.error("Failed to fetch evaluation data", err);
          setEvalError((err as { response?: { data?: { detail?: string } } }).response?.data?.detail || "Failed to fetch evaluation data");
      } finally {
          setIsEvalLoading(false);
      }
  };

  const calculateConfusionMatrix = (
      y_true: (string | number)[],
      y_pred: (string | number)[],
      classOrder?: (string | number)[],
  ) => {
      const classes = (classOrder && classOrder.length > 0)
          ? [...classOrder]
          : Array.from(new Set([...y_true, ...y_pred])).sort((a, b) => String(a).localeCompare(String(b)));
      const matrix = classes.map(trueClass => {
          return classes.map(predClass => {
              return y_true.reduce((count: number, t, i) => {
                  const p = y_pred[i];
                  return (String(t) === String(trueClass) && String(p) === String(predClass)) ? count + 1 : count;
              }, 0);
          });
      });
      return { classes, matrix };
  };

    const calculateROC = (y_true: (string | number)[], y_proba: { classes: (string | number)[], labels?: (string | number)[], values: number[][] }, targetClass: string | number) => {
      // y_proba.values is typed as number[][], so it's always truthy if it exists on the type.
      // If the type definition allows undefined, then the check is valid. Assuming strict null checks.
      // Based on the error "Unnecessary conditional, value is always truthy", y_proba.values is not optional in the type definition above.
      
      const targetClassStr = String(targetClass);
      
      // Find index of target class (normalize types because <select> values are strings)
            const labelList = y_proba.labels && y_proba.labels.length === y_proba.classes.length
                ? y_proba.labels
                : undefined;
            const classIndex = (labelList ?? y_proba.classes).findIndex(c => String(c) === targetClassStr);
      if (classIndex === -1) return null;
      
      const scores = y_proba.values.map(v => v[classIndex] ?? 0);
      
      const data = scores.map((score, i) => ({
          score,
          actual: String(y_true[i]) === targetClassStr ? 1 : 0
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
          if (data[i]!.actual === 1) tp++;
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
              void fetchEvaluationData(selectedJobIds[0]!);
          } else if (evalJobId && !selectedJobIds.includes(evalJobId) && selectedJobIds.length > 0) {
              // If current eval job is deselected, switch to another
              void fetchEvaluationData(selectedJobIds[0]!);
          } else if (selectedJobIds.length === 0) {
              setEvaluationData(null);
              setEvalJobId(null);
          }
      }
  }, [activeView, selectedJobIds, evalJobId]);

  const filteredJobs = jobs.filter(job => {
    const typeMatch = filterType === 'all' || job.job_type === filterType;
    const datasetMatch = selectedDatasetId === 'all' || job.dataset_id === selectedDatasetId;
    const statusMatch = job.status === 'completed';
    return typeMatch && datasetMatch && statusMatch;
  }).sort((a, b) => {
    // Promoted jobs float to top
    if (a.promoted_at && !b.promoted_at) return -1;
    if (!a.promoted_at && b.promoted_at) return 1;
    return 0;
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
      name: shortRunId(job),
      ...metrics
    };
  });

  // Get all unique metric keys from selected jobs
  const metricKeys = Array.from(new Set(
    selectedJobs.flatMap(job => {
        const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
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
      if (key.startsWith('cv_') && !showCvMetrics) return false;
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
      metricGroups.get(base)?.push(key);
  });

  metricGroups.forEach((keys, base) => {
      metricGroups.set(base, keys.sort());
  });

  const availableMetrics = Array.from(metricGroups.keys()).sort();
  const activeMetric = (selectedMetric && availableMetrics.includes(selectedMetric)) 
      ? selectedMetric 
      : availableMetrics[0] || null;

  // const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F'];

  // Feature Importances across selected jobs
  const featureImportancesByJob = selectedJobs.map(job => {
    const result = (job.result ?? {}) as Record<string, unknown>;
    const metrics = result.metrics as Record<string, unknown> | undefined;
    const raw = (metrics?.feature_importances ?? result.feature_importances) as Record<string, number> | undefined;
    return { jobId: job.job_id, modelType: job.model_type ?? 'unknown', importances: raw ?? null };
  });
  const hasFeatureImportances = featureImportancesByJob.some(j => j.importances !== null);

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
             onChange={(e) => { setSelectedDatasetId(e.target.value); }}
           >
             <option value="all">All Datasets</option>
             {datasets.map(ds => (
               <option key={ds.id} value={ds.id}>{ds.name}</option>
             ))}
           </select>
           <select 
             className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
             value={filterType}
             onChange={(e) => { setFilterType(e.target.value as 'all' | 'basic_training' | 'advanced_tuning'); }}
           >
             <option value="all">All Experiments</option>
             <option value="advanced_tuning">Advanced Training</option>
             <option value="basic_training">Standard Training</option>
           </select>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar List */}
        <div className={`${isSidebarCollapsed ? 'w-12' : 'w-80'} border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col relative`}>
          <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex justify-between items-center h-[41px]">
            {!isSidebarCollapsed && (
              <span className="text-xs font-medium text-gray-500 uppercase truncate">
                Select Runs ({selectedJobIds.length})
              </span>
            )}
            <button 
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className={`p-1.5 rounded-lg transition-all duration-200 ${
                  isSidebarCollapsed 
                    ? 'mx-auto text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-gray-100' 
                    : 'ml-auto text-gray-600 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 shadow-sm hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
              title={isSidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            >
              {isSidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
            </button>
          </div>
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
            {filteredJobs.map(job => (
              <div 
                key={job.job_id}
                {...clickableProps(() => { toggleJobSelection(job.job_id); })}
                className={`border-b border-gray-100 dark:border-gray-700 cursor-pointer transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 ${
                  selectedJobIds.includes(job.job_id) ? 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-l-blue-500' : 'border-l-4 border-l-transparent'
                } ${isSidebarCollapsed ? 'p-2 flex justify-center' : 'p-3'}`}
                title={isSidebarCollapsed ? `${shortRunId(job)} · ${job.model_type}` : undefined}
              >
                {isSidebarCollapsed ? (
                    <div className={`w-2 h-2 rounded-full ${
                        job.status === 'completed' ? 'bg-green-500' :
                        job.status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
                    }`} />
                ) : (
                  <>
                <div className="flex justify-between items-start mb-1">
                  <span className="font-mono text-xs font-semibold text-gray-700 dark:text-gray-300 break-all">
                    {shortRunId(job)}
                  </span>
                  <div className="flex items-center gap-2">
                    {job.status === 'completed' && (job.job_type === 'basic_training' || job.job_type === 'advanced_tuning') && (
                      <>
                        <button 
                            onClick={(e) => { void handlePromote(e, job); }}
                            className={`p-1 rounded transition-colors ${
                              job.promoted_at 
                                ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400' 
                                : 'hover:bg-amber-100 dark:hover:bg-amber-900/20 text-gray-400 dark:text-gray-500'
                            }`}
                            title={job.promoted_at ? "Unpromote" : "Promote as Winner"}
                        >
                            <Trophy className="w-3 h-3" />
                        </button>
                        <button 
                            onClick={(e) => { void handleDeploy(e, job.job_id); }}
                            className="p-1 hover:bg-blue-100 dark:hover:bg-blue-900 rounded text-blue-600 dark:text-blue-400"
                            title="Deploy to Test"
                        >
                            <Rocket className="w-3 h-3" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {job.model_type} • {job.dataset_name || 'Unknown Dataset'}
                  {job.job_type === 'advanced_tuning' && (job.search_strategy || (job.config as { tuning?: { strategy?: string } }).tuning?.strategy) && (
                      <span className="ml-1 text-gray-400">
                          ({job.search_strategy || (job.config as { tuning?: { strategy?: string } }).tuning?.strategy})
                      </span>
                  )}
                  {job.branch_index != null && (
                    <span className="ml-1.5 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-[10px] font-semibold">
                      <GitBranch className="w-2.5 h-2.5" /> path {String.fromCharCode(65 + (job.branch_index ?? 0))}
                    </span>
                  )}
                  {job.promoted_at && (
                    <span className="ml-1.5 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-semibold">
                      <Trophy className="w-2.5 h-2.5" /> Winner
                    </span>
                  )}
                </div>
                <div className="flex justify-between items-center text-[10px] text-gray-400">
                  <span>{new Date(job.start_time || job.created_at).toLocaleString()}</span>
                  <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-gray-600 dark:text-gray-300">
                    {getDuration(job.start_time, job.end_time)}
                  </span>
                </div>
                </>
                )}
              </div>
            ))}
            
            {hasMore && (
                <div className="p-3 border-t border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30">
                    <button 
                        onClick={() => loadMoreJobs()}
                        disabled={isLoading}
                        className={`w-full py-2 text-xs font-medium rounded-lg border shadow-sm transition-all duration-200 flex items-center justify-center gap-2 ${
                            isSidebarCollapsed 
                                ? 'bg-transparent border-transparent text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800' 
                                : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:border-blue-400 dark:hover:border-blue-500 hover:text-blue-600 dark:hover:text-blue-400'
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                        title="Load More Runs"
                    >
                        {isLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <ChevronDown className="w-3.5 h-3.5" />}
                        {!isSidebarCollapsed && "Load More Runs"}
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
                      onClick={() => { setActiveView('charts'); }}
                  >
                      Visual Comparison
                  </button>
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'table'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => { setActiveView('table'); }}
                  >
                      Detailed Metrics & Params
                  </button>
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'evaluation'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => { setActiveView('evaluation'); }}
                  >
                      Model Evaluation
                  </button>
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'diff'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => { setActiveView('diff'); }}
                      data-testid="experiments-tab-diff"
                  >
                      Pipeline Diff
                  </button>
                  {hasFeatureImportances && (
                  <button
                      className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          activeView === 'importance'
                              ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                              : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                      }`}
                      onClick={() => { setActiveView('importance'); }}
                  >
                      Feature Importance
                  </button>
                  )}
              </div>

              {/* Branch Comparison Card — shown when selected jobs share a parallel run */}
              {(() => {
                const branchJobs = selectedJobs.filter(j => j.parent_pipeline_id != null);
                // Group by parent_pipeline_id
                const groups = new Map<string, typeof selectedJobs>();
                branchJobs.forEach(j => {
                  const key = j.parent_pipeline_id!;
                  if (!groups.has(key)) groups.set(key, []);
                  groups.get(key)!.push(j);
                });
                // Only show groups with 2+ branches
                const multiGroups = Array.from(groups.entries()).filter(([, jobs]) => jobs.length >= 2);
                if (multiGroups.length === 0) return null;

                // Collect all metric keys across branch jobs
                const allBranchMetricKeys = Array.from(new Set(
                  branchJobs.flatMap(j => {
                    const m = (j.metrics || j.result?.metrics || {}) as Record<string, unknown>;
                    return Object.keys(m).filter(k => typeof m[k] === 'number');
                  })
                )).filter(k => k.startsWith('test_') || k === 'best_score').sort();

                return multiGroups.map(([parentId, groupJobs]) => (
                  <div key={parentId} className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/10 dark:to-blue-900/10 rounded-lg border border-purple-200 dark:border-purple-800 p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <GitBranch className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                      <h3 className="text-sm font-semibold text-purple-800 dark:text-purple-300">
                        Parallel Run Comparison
                      </h3>
                      <span className="text-xs text-purple-500 dark:text-purple-400 font-mono">
                        {groupJobs.length} paths · run {parentId.replace(/^preview_/, '').replace(/__branch_.*$/, '').slice(0, 8)}
                      </span>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="border-b border-purple-200 dark:border-purple-700">
                            <th className="px-3 py-1.5 text-gray-600 dark:text-gray-400 font-medium">Metric</th>
                            {groupJobs.sort((a, b) => (a.branch_index ?? 0) - (b.branch_index ?? 0)).map(j => (
                              <th key={j.job_id} className="px-3 py-1.5 font-medium text-gray-700 dark:text-gray-300">
                                <div className="flex items-center gap-1">
                                  <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: `hsl(${(j.branch_index ?? 0) * 120}, 70%, 50%)` }} />
                                  Path {String.fromCharCode(65 + (j.branch_index ?? 0))} · {j.model_type}
                                  {j.promoted_at && <Trophy className="w-3 h-3 text-amber-500 ml-1" />}
                                </div>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {allBranchMetricKeys.map(key => {
                            // Find best value for highlighting
                            const values = groupJobs.map(j => {
                              const m = (j.metrics || j.result?.metrics || {}) as Record<string, number>;
                              return m[key];
                            });
                            const isLowerBetter = key.includes('loss') || key.includes('error') || key.includes('mse') || key.includes('mae');
                            const bestVal = isLowerBetter
                              ? Math.min(...values.filter(v => v != null))
                              : Math.max(...values.filter(v => v != null));

                            return (
                              <tr key={key} className="border-b border-purple-100 dark:border-purple-800/50">
                                <td className="px-3 py-1.5 text-gray-600 dark:text-gray-400">
                                  {key === 'best_score'
                                    ? `Best Score (${formatMetricName(getJobScoringMetric(groupJobs[0] ?? {})) || 'CV'})`
                                    : key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                </td>
                                {groupJobs.map(j => {
                                  const m = (j.metrics || j.result?.metrics || {}) as Record<string, number>;
                                  const val = m[key];
                                  const isBest = val != null && val === bestVal;
                                  return (
                                    <td key={j.job_id} className={`px-3 py-1.5 font-mono ${isBest ? 'text-green-600 dark:text-green-400 font-bold' : 'text-gray-600 dark:text-gray-300'}`}>
                                      {val != null ? val.toFixed(4) : '-'}
                                      {isBest && ' ★'}
                                    </td>
                                  );
                                })}
                              </tr>
                            );
                          })}
                          <tr className="border-b border-purple-100 dark:border-purple-800/50">
                            <td className="px-3 py-1.5 text-gray-600 dark:text-gray-400">Duration</td>
                            {groupJobs.map(j => (
                              <td key={j.job_id} className="px-3 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                                {getDuration(j.start_time, j.end_time)}
                              </td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                ));
              })()}

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
                                onChange={e => { setShowTrainMetrics(e.target.checked); }}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Train</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showTestMetrics} 
                                onChange={e => { setShowTestMetrics(e.target.checked); }}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Test</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showValMetrics} 
                                onChange={e => { setShowValMetrics(e.target.checked); }}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Validation</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input 
                                type="checkbox" 
                                checked={showCvMetrics} 
                                onChange={e => { setShowCvMetrics(e.target.checked); }}
                                className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                            />
                            <span className="text-gray-700 dark:text-gray-300">Cross-Validation</span>
                        </label>
                    </div>
                </div>

                {/* Metric Selector Tabs */}
                {availableMetrics.length > 0 && (
                    <div className="flex flex-wrap gap-2 border-b border-gray-200 dark:border-gray-700 pb-2">
                        {availableMetrics.map(metric => (
                            <button
                                key={metric}
                                onClick={() => { setSelectedMetric(metric); }}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                                    activeMetric === metric
                                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                                }`}
                            >
                                {metric.toUpperCase()}
                                {getMetricDescription(metric) && <InfoTooltip size="sm" text={getMetricDescription(metric)!} />}
                            </button>
                        ))}
                    </div>
                )}

                {activeMetric && metricGroups.has(activeMetric) && (
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                        <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={metricsData} margin={{ top: 5, right: 30, left: 30, bottom: 40 }}>
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
                            {metricGroups.get(activeMetric)?.map((key) => {
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
                          <th key={job.job_id} className="px-4 py-2 font-mono break-all min-w-[100px]">
                            {shortRunId(job)}
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
                            <div className="flex items-center gap-1.5">
                              {job.model_type}
                              {job.branch_index != null && (
                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-[10px] font-semibold">
                                  <GitBranch className="w-2.5 h-2.5" /> Path {String.fromCharCode(65 + (job.branch_index ?? 0))}
                                </span>
                              )}
                              {job.promoted_at && (
                                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-semibold">
                                  <Trophy className="w-2.5 h-2.5" /> Winner
                                </span>
                              )}
                            </div>
                          </td>
                        ))}
                      </tr>
                      {/* Pipeline Steps — preprocessing/splits/etc that fed each terminal */}
                      <tr
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => { setIsPipelineExpanded(!isPipelineExpanded); }}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isPipelineExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Pipeline Steps
                        </td>
                      </tr>
                      {isPipelineExpanded && (() => {
                        // Walk each job's graph backwards from the training
                        // terminal to collect preprocessing / split / encoding
                        // ancestors. Each row in the table is one position in
                        // the chain (Step 1, Step 2, …) so users can compare
                        // what came before each model side-by-side.
                        type GraphNode = { node_id: string; step_type?: string; params?: Record<string, unknown>; inputs?: string[] };
                        const friendlyStep = (st: string): string => {
                          if (!st) return '';
                          if (st.includes('_')) {
                            return st.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                          }
                          return st.replace(/(?<!^)(?=[A-Z])/g, ' ').trim();
                        };
                        const collectChain = (job: typeof selectedJobs[number]): GraphNode[] => {
                          const graphNodes = (job.graph?.nodes as GraphNode[] | undefined) || [];
                          if (graphNodes.length === 0 || !job.node_id) return [];
                          const map = new Map(graphNodes.map(n => [n.node_id, n]));
                          // BFS backwards from the terminal, then return in
                          // root → terminal order (excluding the terminal).
                          const seen = new Set<string>();
                          const order: GraphNode[] = [];
                          const walk = (id: string): void => {
                            if (seen.has(id)) return;
                            seen.add(id);
                            const n = map.get(id);
                            if (!n) return;
                            for (const parent of n.inputs || []) walk(parent);
                            order.push(n);
                          };
                          walk(job.node_id);
                          return order.filter(n => n.node_id !== job.node_id);
                        };
                        const summarizeStep = (n: GraphNode): string => {
                          const display = (n.params?._display_name as string | undefined) || friendlyStep(String(n.step_type || ''));
                          const interestingKeys = ['method', 'strategy', 'columns', 'target_column', 'test_size', 'val_size', 'random_state', 'n_neighbors'];
                          const detail: string[] = [];
                          for (const k of interestingKeys) {
                            const v = n.params?.[k];
                            if (v === undefined || v === null || v === '') continue;
                            if (Array.isArray(v) && v.length === 0) continue;
                            let rendered: string;
                            if (Array.isArray(v)) {
                              // Show real column names; truncate long lists so
                              // the cell stays readable.
                              const items = v.map(x => String(x));
                              rendered = items.length > 4
                                ? `[${items.slice(0, 4).join(', ')}, +${items.length - 4} more]`
                                : `[${items.join(', ')}]`;
                            } else if (typeof v === 'object') {
                              rendered = JSON.stringify(v);
                            } else {
                              rendered = String(v);
                            }
                            detail.push(`${k}=${rendered}`);
                          }
                          // Special-case Feature Generation: its config lives
                          // under `operations: MathOperation[]`, not the
                          // generic keys above. Render the per-op summary
                          // (e.g. `multiply(a, b)`, `month(date)`) so the
                          // comparison row carries real signal instead of
                          // just "Feature Generation".
                          const ops = n.params?.['operations'];
                          if (Array.isArray(ops) && ops.length > 0) {
                            const formatted = ops.slice(0, 3).map((raw) => {
                              const op = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
                              const method = String(op['method'] ?? op['operation_type'] ?? 'op');
                              const inputs = Array.isArray(op['input_columns']) ? op['input_columns'] as unknown[] : [];
                              const secondary = Array.isArray(op['secondary_columns']) ? op['secondary_columns'] as unknown[] : [];
                              const operands = [...inputs, ...secondary].map(String);
                              const args = operands.length > 2
                                ? `${operands.slice(0, 2).join(', ')}, +${operands.length - 2}`
                                : operands.join(', ');
                              return args ? `${method}(${args})` : method;
                            });
                            const tail = ops.length > 3 ? `, +${ops.length - 3} more` : '';
                            detail.push(`ops=[${formatted.join(', ')}${tail}]`);
                          }
                          return detail.length > 0 ? `${display} (${detail.join(', ')})` : display;
                        };
                        const chainsByJob = new Map(selectedJobs.map(j => [j.job_id, collectChain(j)]));
                        const allChains = Array.from(chainsByJob.values());
                        if (allChains.every(c => c.length === 0)) {
                          return (
                            <tr className="bg-white dark:bg-gray-800">
                              <td className="px-4 py-1.5 text-gray-400 italic pl-8" colSpan={selectedJobs.length + 1}>
                                No upstream pipeline steps captured for these runs.
                              </td>
                            </tr>
                          );
                        }
                        // L6 — Align rows by node_id rather than by raw
                        // chain index. When the trunk is shared (the
                        // common case after the per-branch graph snapshot
                        // fix), shared nodes occupy a single row across
                        // all columns. Branch-only steps land on their
                        // own rows with em-dashes in the other columns.
                        // Algorithm: walk every chain in lockstep,
                        // emitting the next un-emitted node id in chain
                        // order. A node id appears in the merged order
                        // the first time any chain references it.
                        const mergedOrder: string[] = [];
                        const emitted = new Set<string>();
                        const cursors = allChains.map(() => 0);
                        // Bound the loop to the sum of chain lengths to
                        // guarantee termination on pathological inputs.
                        const safety = allChains.reduce((s, c) => s + c.length, 0) + 1;
                        for (let guard = 0; guard < safety; guard++) {
                          let advanced = false;
                          for (let ci = 0; ci < allChains.length; ci++) {
                            const chain = allChains[ci];
                            if (!chain) continue;
                            // Skip past nodes we've already emitted.
                            while (cursors[ci]! < chain.length && emitted.has(chain[cursors[ci]!]!.node_id)) {
                              cursors[ci] = cursors[ci]! + 1;
                            }
                            if (cursors[ci]! < chain.length) {
                              const candidate = chain[cursors[ci]!]!;
                              if (!emitted.has(candidate.node_id)) {
                                mergedOrder.push(candidate.node_id);
                                emitted.add(candidate.node_id);
                                advanced = true;
                                break; // restart sweep so other chains catch up
                              }
                            }
                          }
                          if (!advanced) break;
                        }
                        return mergedOrder.map((nid, idx) => {
                          // Per-row cell strings, plus a sameness flag
                          // for shared trunk highlighting.
                          const cells = selectedJobs.map(job => {
                            const chain = chainsByJob.get(job.job_id) || [];
                            const step = chain.find(n => n.node_id === nid);
                            return step ? summarizeStep(step) : null;
                          });
                          const presentCells = cells.filter((c): c is string => c !== null);
                          const allPresent = presentCells.length === cells.length;
                          const allSame = allPresent && presentCells.every(c => c === presentCells[0]);
                          // Shared trunk row → muted background; divergent
                          // row (some columns dash, others differ) →
                          // amber accent so the diff jumps out.
                          const rowTone = allSame
                            ? 'bg-white dark:bg-gray-800'
                            : 'bg-amber-50/40 dark:bg-amber-900/10';
                          return (
                            <tr key={`pipeline-step-${nid}`} className={`${rowTone} hover:bg-gray-50 dark:hover:bg-gray-700/50`}>
                              <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">Step {idx + 1}</td>
                              {cells.map((text, ci) => (
                                <td
                                  key={selectedJobs[ci]?.job_id ?? ci}
                                  className={
                                    text === null
                                      ? 'px-4 py-1.5 text-gray-300 dark:text-gray-600'
                                      : allSame
                                        ? 'px-4 py-1.5 text-gray-500 dark:text-gray-400'
                                        : 'px-4 py-1.5 text-gray-900 dark:text-gray-100 font-medium'
                                  }
                                >
                                  {text ?? <span className="text-gray-400">—</span>}
                                </td>
                              ))}
                            </tr>
                          );
                        });
                      })()}
                      {/* Metrics Section in Table */}
                      <tr 
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => { setIsMetricsExpanded(!isMetricsExpanded); }}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isMetricsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Key Metrics
                        </td>
                      </tr>
                      {isMetricsExpanded && metricKeys.map(metricKey => (
                        <tr key={metricKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                            <div className="flex items-center gap-1">
                              {metricKey === 'best_score'
                                ? `Best Score (${formatMetricName(getJobScoringMetric(selectedJobs[0] ?? {})) || 'CV'})`
                                : metricKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                              {getMetricDescription(metricKey) && <InfoTooltip size="sm" text={getMetricDescription(metricKey)!} />}
                            </div>
                          </td>
                          {selectedJobs.map(job => {
                             const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
                             const val = m[metricKey];
                             return (
                                <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                                  {typeof val === 'number' ? (metricKey.endsWith('_std') ? val.toFixed(6) : val.toFixed(4)) : '-'}
                                </td>
                             );
                          })}
                        </tr>
                      ))}
                      {/* Hyperparameters (actual model params only) */}
                      <tr 
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => { setIsParamsExpanded(!isParamsExpanded); }}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isParamsExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Hyperparameters
                        </td>
                      </tr>
                      {/* Extract actual model hyperparameters: best_params for advanced, nested hyperparameters for basic */}
                      {isParamsExpanded && (() => {
                        const getModelParams = (job: { job_type?: string; hyperparameters?: unknown }): Record<string, unknown> => {
                          const hp = job.hyperparameters as Record<string, unknown> | undefined;
                          if (!hp) return {};
                          if (job.job_type === 'advanced_tuning') {
                            // For advanced tuning, hyperparameters IS the best_params (or search_space) directly
                            return hp;
                          }
                          // Basic training: extract the nested 'hyperparameters' dict (actual model params)
                          const nested = hp.hyperparameters;
                          if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
                            return nested as Record<string, unknown>;
                          }
                          return {};
                        };
                        const allKeys = Array.from(new Set(selectedJobs.flatMap(job => Object.keys(getModelParams(job)))));
                        if (allKeys.length === 0) {
                          return (
                            <tr className="bg-white dark:bg-gray-800">
                              <td className="px-4 py-1.5 text-gray-400 dark:text-gray-500 pl-8 italic" colSpan={selectedJobs.length + 1}>
                                Default parameters (none customized)
                              </td>
                            </tr>
                          );
                        }
                        return allKeys.map(paramKey => (
                          <tr key={paramKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                            <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                              <div className="flex items-center gap-1">
                                {paramKey}
                                {getHyperparamDescription(paramKey) && <InfoTooltip size="sm" text={getHyperparamDescription(paramKey)!} />}
                              </div>
                            </td>
                            {selectedJobs.map(job => {
                              const params = getModelParams(job);
                              const val = params[paramKey];
                              return (
                                <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
                                  {val === undefined ? '-' : typeof val === 'object' ? JSON.stringify(val) : String(val)}
                                </td>
                              );
                            })}
                          </tr>
                        ));
                      })()}

                      {/* Training Configuration */}
                      <tr 
                        className="bg-gray-50/50 dark:bg-gray-900/20 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                        onClick={() => { setIsTuningExpanded(!isTuningExpanded); }}
                      >
                        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2" colSpan={selectedJobs.length + 1}>
                          {isTuningExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          Training Configuration
                        </td>
                      </tr>
                      {isTuningExpanded && (
                        <>
                          {['Target Column', 'CV Enabled', 'CV Method', 'CV Folds', 'CV Shuffle', 'CV Random State', ...(selectedJobs.some(j => j.job_type === 'advanced_tuning') ? ['Strategy', 'Strategy Params', 'Metric', 'Trials'] : [])].map(field => (
                            <tr key={field} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                              <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
                              <div className="flex items-center gap-1">
                                {field}
                                {getTrainingConfigDescription(field) && <InfoTooltip size="sm" text={getTrainingConfigDescription(field)!} />}
                              </div>
                            </td>
                              {selectedJobs.map(job => {
                                // Resolve config: for advanced tuning use job.config or graph node params,
                                // for basic training use job.hyperparameters (which contains full node params)
                                let cfg: Record<string, unknown> | null = null;
                                if (job.job_type === 'advanced_tuning') {
                                  const nodeParams = (job.config as Record<string, unknown>) ||
                                    (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id)?.params;
                                  cfg = nodeParams || null;
                                } else {
                                  cfg = (job.hyperparameters as Record<string, unknown>) ||
                                    (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id)?.params || null;
                                }

                                if (!cfg) {
                                  return <td key={job.job_id} className="px-4 py-1.5 text-gray-400">-</td>;
                                }

                                // For advanced tuning, CV params are inside tuning_config
                                const tuningConfig = cfg.tuning_config as Record<string, unknown> | undefined;
                                const cvSource = (job.job_type === 'advanced_tuning' && tuningConfig ? tuningConfig : cfg) as Record<string, unknown>;
                                // Local helper: coerce unknown-typed config field to a renderable scalar.
                                const str = (v: unknown, fallback: string | number = '-'): string | number =>
                                  v === undefined || v === null || v === '' ? fallback : (typeof v === 'number' ? v : String(v));

                                let value: string | number = '-';
                                if (field === 'Target Column') value = str(cfg.target_column ?? job.target_column);
                                if (field === 'CV Enabled') value = cvSource.cv_enabled ? 'Yes' : 'No';
                                if (field === 'CV Method') value = cvSource.cv_enabled ? str(cvSource.cv_type, 'Unknown') : '-';
                                if (field === 'CV Folds') value = cvSource.cv_enabled ? str(cvSource.cv_folds) : '-';
                                if (field === 'CV Shuffle') value = cvSource.cv_enabled ? (cvSource.cv_shuffle ? 'Yes' : 'No') : '-';
                                if (field === 'CV Random State') value = cvSource.cv_enabled ? str(cvSource.cv_random_state) : '-';
                                if (field === 'Strategy') value = str(tuningConfig?.strategy ?? tuningConfig?.search_strategy);
                                if (field === 'Strategy Params') {
                                  const sp = tuningConfig?.strategy_params as Record<string, unknown> | undefined;
                                  const strategy = String(tuningConfig?.strategy ?? tuningConfig?.search_strategy ?? '');
                                  if (sp && Object.keys(sp).length > 0) {
                                    value = JSON.stringify(sp);
                                  } else if (strategy === 'optuna') {
                                    value = 'sampler: tpe · pruner: median (defaults)';
                                  } else if (strategy === 'halving_grid' || strategy === 'halving_random') {
                                    value = 'factor: 3 · min: exhaust (defaults)';
                                  } else {
                                    value = '-';
                                  }
                                }
                                if (field === 'Metric') value = str(tuningConfig?.metric);
                                if (field === 'Trials') value = str(tuningConfig?.n_trials);

                                return (
                                  <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300 capitalize">
                                    {value}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </>
                      )}
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
                                    onClick={() => { void fetchEvaluationData(id); }}
                                    className={`px-3 py-1 text-xs font-mono rounded border whitespace-nowrap ${
                                        evalJobId === id 
                                            ? 'bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300'
                                            : 'bg-white border-gray-200 text-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400'
                                    }`}
                                >
                                    {id}
                                </button>
                            ))}
                        </div>
                    )}

                    {isEvalLoading ? (
                        <div className="h-64 flex items-center justify-center">
                            <LoadingState message="Loading evaluation data..." />
                        </div>
                    ) : evalError ? (
                        <div className="h-64 flex items-center justify-center">
                            <ErrorState error={evalError} />
                        </div>
                    ) : !evaluationData ? (
                        <div className="h-64 flex items-center justify-center text-gray-400 italic text-center">
                            <p>Select a completed job to view evaluation details.</p>
                            <p className="text-xs mt-2 opacity-70">(Note: Only jobs run after this update have evaluation artifacts)</p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {/* Controls for Evaluation View — sticky so it stays visible while scrolling splits */}
                            <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                {/* Split visibility toggles */}
                                <div className="flex items-center gap-1 text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide">Splits:</div>
                                <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                                    <input type="checkbox" checked={showTrainMetrics} onChange={e => { setShowTrainMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                                    <span className="text-gray-700 dark:text-gray-300">Train</span>
                                </label>
                                <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                                    <input type="checkbox" checked={showTestMetrics} onChange={e => { setShowTestMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                                    <span className="text-gray-700 dark:text-gray-300">Test</span>
                                </label>
                                <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                                    <input type="checkbox" checked={showValMetrics} onChange={e => { setShowValMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                                    <span className="text-gray-700 dark:text-gray-300">Validation</span>
                                </label>

                                {/* Classification controls */}
                                {evaluationData.problem_type === 'classification' && evaluationData.splits.train?.y_proba && (() => {
                                    const proba = evaluationData.splits.train.y_proba!;
                                    const isBinary = proba.classes.length === 2;
                                    return (
                                        <>
                                            <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                                            {/* Class selector — hidden for binary: both classes always shown inline */}
                                            {!isBinary && (
                                                <div className="flex items-center gap-2">
                                                    <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Class:</span>
                                                    <select
                                                        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
                                                        value={selectedRocClass || ''}
                                                        onChange={(e) => { setSelectedRocClass(e.target.value); }}
                                                    >
                                                        {proba.classes.map((c: string | number, idx: number) => {
                                                            const label = proba.labels?.[idx] ?? c;
                                                            return <option key={String(c)} value={String(label)}>{String(label)}</option>;
                                                        })}
                                                    </select>
                                                </div>
                                            )}
                                            <div className="flex items-center gap-2">
                                                <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Threshold:</span>
                                                <input
                                                    type="range" min={0.01} max={0.99} step={0.01}
                                                    value={threshold}
                                                    onChange={(e) => { setThreshold(parseFloat(e.target.value)); }}
                                                    className="w-28 accent-blue-500"
                                                />
                                                <span className="text-sm font-mono font-semibold text-blue-600 dark:text-blue-400 w-9">{threshold.toFixed(2)}</span>
                                            </div>
                                            {proba.labels && proba.labels.length === proba.classes.length && (
                                                <div className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                                                    ({proba.classes.map((c, idx) => `${String(c)}→${String(proba.labels?.[idx] ?? c)}`).join(', ')})
                                                </div>
                                            )}
                                            {/* Overall / Per Class toggle — hidden for binary: no separate tab needed */}
                                            {!isBinary && (
                                                <>
                                                    <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                                                    <div className="flex items-center rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-xs font-medium">
                                                        <button onClick={() => setCmView('overall')} className={`px-3 py-1.5 transition-colors ${cmView === 'overall' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Overall</button>
                                                        <button onClick={() => setCmView('per-class')} className={`px-3 py-1.5 transition-colors border-l border-gray-200 dark:border-gray-700 ${cmView === 'per-class' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Per Class</button>
                                                    </div>
                                                </>
                                            )}
                                        </>
                                    );
                                })()}
                            </div>

                            {(evaluationData.problem_type === 'regression' || cmView === 'overall' || evaluationData.splits.train?.y_proba?.classes.length === 2) && (
                            <div className="flex flex-col gap-6">
                            {/* Render charts for each split */}
                            {Object.entries(evaluationData.splits)
                                .filter(([splitName]) => {
                                    if (splitName === 'train' && !showTrainMetrics) return false;
                                    if (splitName === 'test' && !showTestMetrics) return false;
                                    if (splitName === 'validation' && !showValMetrics) return false;
                                    return true;
                                })
                                .map(([splitName, splitData]: [string, EvaluationSplit]) => {
                                const data = splitData.y_true.map((y: unknown, i: number) => ({
                                    x: y,
                                    y: splitData.y_pred[i]
                                }));
                                
                                return (
                                    <div key={splitName} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 capitalize">{splitName} Set</h4>
                                        
                                        {evaluationData.problem_type === 'regression' ? (
                                            <div className="grid grid-cols-1 gap-8">
                                                {/* Scatter Plot: Actual vs Predicted */}
                                                <div className="h-[300px] relative group" id={`${splitName}-actual-pred`}>
                                                    <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                       <button 
                                                         onClick={() => void handleDownload(`${splitName}-actual-pred`, `${splitName}_actual_vs_predicted`)}
                                                         disabled={downloadingChart === `${splitName}-actual-pred`}
                                                         className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                         title="Download Graph"
                                                       >
                                                          {downloadingChart === `${splitName}-actual-pred` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-actual-pred` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                       </button>
                                                    </div>
                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center">Actual vs Predicted</h5>
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 30 }}>
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
                                                                label={{ value: 'Predicted Values', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 12 }} 
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
                                                <div className="h-[300px] relative group" id={`${splitName}-residuals`}>
                                                    <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                       <button 
                                                         onClick={() => void handleDownload(`${splitName}-residuals`, `${splitName}_residuals`)}
                                                         disabled={downloadingChart === `${splitName}-residuals`}
                                                         className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                         title="Download Graph"
                                                       >
                                                          {downloadingChart === `${splitName}-residuals` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-residuals` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                       </button>
                                                    </div>
                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 text-center">Residuals vs Predicted</h5>
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 30 }}>
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
                                                                label={{ value: 'Residuals (Actual - Predicted)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 12 }} 
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
                                                                data={data.map((d: unknown) => {
                                                                    const val = d as { x: number; y: number };
                                                                    return { ...val, residual: val.x - val.y };
                                                                })} 
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
                                                <div className="flex flex-col items-center justify-center relative group" id={`${splitName}-confusion-matrix`}>
                                                    <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                       <button 
                                                         onClick={() => void handleDownload(`${splitName}-confusion-matrix`, `${splitName}_confusion_matrix`)}
                                                         disabled={downloadingChart === `${splitName}-confusion-matrix`}
                                                         className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                         title="Download Graph"
                                                       >
                                                          {downloadingChart === `${splitName}-confusion-matrix` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-confusion-matrix` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                       </button>
                                                    </div>
                                                    {(() => {
                                                        const proba = splitData.y_proba;
                                                        const classOrder = proba?.classes;
                                                        let yTrueForCm: (string | number)[] = splitData.y_true;
                                                        let yPredForCm: (string | number)[] = splitData.y_pred;
                                                        if (proba?.labels && proba.labels.length === proba.classes.length) {
                                                            const labelToClass = new Map<string, string | number>();
                                                            proba.labels.forEach((label, idx) => {
                                                                const cls = proba.classes[idx];
                                                                if (cls !== undefined) labelToClass.set(String(label), cls);
                                                            });
                                                            yTrueForCm = splitData.y_true.map(y => labelToClass.get(String(y)) ?? y);
                                                            yPredForCm = splitData.y_pred.map(y => labelToClass.get(String(y)) ?? y);
                                                        }

                                                        // Apply OvR threshold for the selected class (works for binary and multiclass)
                                                        if (proba && selectedRocClass) {
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const posIdx = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (posIdx !== -1) {
                                                                const posVal = proba.classes[posIdx];
                                                                const origPred = [...yPredForCm];
                                                                if (posVal !== undefined) {
                                                                    yPredForCm = proba.values.map((v, i) => {
                                                                        if ((v[posIdx] ?? 0) >= threshold) return posVal;
                                                                        // Argmax of all other classes
                                                                        let bestIdx = -1, bestProb = -Infinity;
                                                                        v.forEach((p, idx) => {
                                                                            if (idx !== posIdx && p > bestProb) { bestProb = p; bestIdx = idx; }
                                                                        });
                                                                        return bestIdx >= 0 ? (proba.classes[bestIdx] ?? origPred[i]!) : (origPred[i]!);
                                                                    });
                                                                }
                                                            }
                                                        }

                                                        const { classes, matrix } = calculateConfusionMatrix(yTrueForCm, yPredForCm, classOrder);

                                                        // Compute live OvR metrics for the selected class
                                                        let liveMetrics: { accuracy: number; precision: number; recall: number; f1: number } | null = null;
                                                        if (selectedRocClass && proba) {
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const posClassIdx = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (posClassIdx !== -1) {
                                                                const posVal = proba.classes[posClassIdx];
                                                                const posMatrixIdx = classes.findIndex(c => String(c) === String(posVal));
                                                                if (posMatrixIdx !== -1) {
                                                                    const tp = matrix[posMatrixIdx]?.[posMatrixIdx] ?? 0;
                                                                    const fp = matrix.reduce((s, row, ri) => ri !== posMatrixIdx ? s + (row[posMatrixIdx] ?? 0) : s, 0);
                                                                    const fn = (matrix[posMatrixIdx] ?? []).reduce((s, v, ci) => ci !== posMatrixIdx ? s + v : s, 0);
                                                                    const total = matrix.flat().reduce((a, b) => a + b, 0);
                                                                    const tn = total - tp - fp - fn;
                                                                    const accuracy = total > 0 ? (tp + tn) / total : 0;
                                                                    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
                                                                    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
                                                                    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
                                                                    liveMetrics = { accuracy, precision, recall, f1 };
                                                                }
                                                            }
                                                        }

                                                        const cellSize = classes.length <= 3 ? 'w-20 h-16' : classes.length <= 5 ? 'w-14 h-12' : 'w-10 h-9';
                                                        const cellText = classes.length <= 5 ? 'text-xs' : 'text-[10px]';
                                                        const cellW = cellSize.split(' ')[0]!;
                                                        const cellH = cellSize.split(' ')[1]!;

                                                        return (
                                                            <div className="flex flex-col items-center w-full">
                                                                {/* ── OVERALL MATRIX — Actual label lives alongside ONLY the data rows ── */}
                                                                <div className="flex flex-col">
                                                                    {/* Predicted header: spacer = row-label (76px) + Actual-label (20px) + gap (4px) = 100px */}
                                                                    <div className="flex items-center mb-1">
                                                                        <div className="w-[100px] shrink-0" />
                                                                        <div className="flex-1 flex items-center justify-center gap-1">
                                                                            <span className="text-[11px] text-gray-400 dark:text-gray-500">Predicted</span>
                                                                            <InfoTooltip text="Columns = what the model predicted. Each column is one class. Read a column down ↓ to see all samples predicted as that class." size="sm" />
                                                                        </div>
                                                                    </div>
                                                                    {/* Col-name headers — same 100px spacer so they sit directly above cells */}
                                                                    <div className="flex mb-0.5">
                                                                        <div className="w-[100px] shrink-0" />
                                                                        {classes.map(c => (
                                                                            <div key={String(c)} className={`${cellW} text-center text-[11px] font-medium text-gray-500 dark:text-gray-400 pb-1 truncate`} title={String(c)}>
                                                                                {String(c)}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                    {/* Body: Actual label sits inside items-stretch so its height = matrix height → perfectly centered */}
                                                                    <div className="flex items-stretch">
                                                                        <div className="flex flex-col items-center justify-center mr-1" style={{ width: '20px' }}>
                                                                            <InfoTooltip text="Rows = actual / true labels. Read a row across → to see where each true class ended up. Green diagonal = correct; red off-diagonal = misclassification." size="sm" />
                                                                            <span className="text-[11px] text-gray-400 dark:text-gray-500" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual</span>
                                                                        </div>
                                                                        <div className="border border-gray-200 dark:border-gray-700 rounded overflow-hidden">
                                                                            {matrix.map((row, i) => {
                                                                                const rowTotal = row.reduce((a, b) => a + b, 0);
                                                                                return (
                                                                                    <div key={i} className="flex">
                                                                                        <div className={`w-[76px] ${cellH} flex items-center justify-end pr-2 text-[11px] font-medium text-gray-500 dark:text-gray-400 truncate border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 shrink-0`} title={String(classes[i])}>
                                                                                            {String(classes[i])}
                                                                                        </div>
                                                                                        {row.map((count, j) => {
                                                                                            const isDiag = i === j;
                                                                                            const intensity = rowTotal > 0 ? count / rowTotal : 0;
                                                                                            const bgColor = isDiag
                                                                                                ? `rgba(34, 197, 94, ${intensity * 0.75 + 0.08})`
                                                                                                : `rgba(239, 68, 68, ${intensity * 0.65 + 0.04})`;
                                                                                            const textColor = intensity > 0.45 ? 'white' : undefined;
                                                                                            const pct = rowTotal > 0 ? ((count / rowTotal) * 100).toFixed(0) : '0';
                                                                                            return (
                                                                                                <div
                                                                                                    key={j}
                                                                                                    className={`${cellSize} flex flex-col items-center justify-center border border-gray-100 dark:border-gray-800 cursor-default`}
                                                                                                    style={{ backgroundColor: bgColor, color: textColor }}
                                                                                                    title={`True: ${classes[i]}, Pred: ${classes[j]}\nCount: ${count}  |  ${pct}% of actual "${classes[i]}"\n${isDiag ? '✓ Correct prediction' : '✗ Misclassification'}`}
                                                                                                >
                                                                                                    <span className={`${cellText} font-mono font-bold leading-none`}>{count}</span>
                                                                                                    <span className="text-[9px] leading-none opacity-75 mt-0.5">{pct}%</span>
                                                                                                </div>
                                                                                            );
                                                                                        })}
                                                                                    </div>
                                                                                );
                                                                            })}
                                                                        </div>
                                                                    </div>
                                                                </div>

                                                                {/* Footer: label + threshold tooltip */}
                                                                <div className="mt-3 flex items-center gap-1.5 text-xs text-gray-400">
                                                                    <span>Confusion Matrix</span>
                                                                    {selectedRocClass && (
                                                                        <>
                                                                            <span className="font-mono text-blue-500 dark:text-blue-400">@ t={threshold.toFixed(2)}</span>
                                                                            <InfoTooltip
                                                                                text={`Threshold rule (≥): a sample is predicted as "${selectedRocClass}" when P("${selectedRocClass}") ≥ ${threshold.toFixed(2)} (equal or above). Otherwise the class with the highest remaining probability wins.\n↑ Raise threshold → fewer positives predicted, lower recall, higher precision.\n↓ Lower threshold → more positives predicted, higher recall, lower precision.\nGreen cells = correct; red cells = errors. Percentages show % of each actual class row.`}
                                                                                align="center"
                                                                            />
                                                                        </>
                                                                    )}
                                                                </div>
                                                                {/* Live metric tiles */}
                                                                {liveMetrics && (
                                                                    <div className="mt-2 grid grid-cols-4 gap-1.5 text-xs w-full">
                                                                        {([
                                                                            { label: 'Accuracy', value: liveMetrics.accuracy, tip: 'Overall fraction of correct predictions (OvR: treats selected class as positive).' },
                                                                            { label: 'Precision', value: liveMetrics.precision, tip: 'Of all samples predicted as this class, how many actually are? High = few false alarms.' },
                                                                            { label: 'Recall', value: liveMetrics.recall, tip: 'Of all actual samples of this class, how many did the model catch? High = few misses.' },
                                                                            { label: 'F1', value: liveMetrics.f1, tip: 'Harmonic mean of Precision and Recall. Balances both — best single metric for imbalanced classes.' },
                                                                        ] as { label: string; value: number; tip: string }[]).map(({ label, value, tip }) => {
                                                                            const color = value >= 0.8 ? 'text-green-600 dark:text-green-400' : value >= 0.6 ? 'text-yellow-500 dark:text-yellow-400' : 'text-red-500 dark:text-red-400';
                                                                            return (
                                                                                <div key={label} className="flex flex-col items-center bg-gray-50 dark:bg-gray-900 rounded px-1.5 py-1.5 gap-0.5">
                                                                                    <div className="flex items-center gap-0.5">
                                                                                        <span className="text-gray-500 dark:text-gray-400">{label}</span>
                                                                                        <InfoTooltip text={tip} size="sm" align="center" />
                                                                                    </div>
                                                                                    <span className={`font-mono font-semibold ${color}`}>{value.toFixed(3)}</span>
                                                                                </div>
                                                                            );
                                                                        })}
                                                                    </div>
                                                                )}
                                                                {/* Binary: show both classes' Prec/Rec/F1 inline — no need to switch to Per Class tab */}
                                                                {classes.length === 2 && (
                                                                    <div className="mt-3 border-t border-gray-100 dark:border-gray-700 pt-3">
                                                                        <div className="flex items-center gap-1 mb-2">
                                                                            <span className="text-[11px] font-medium text-gray-400 dark:text-gray-500">Per Class</span>
                                                                            <InfoTooltip text="Precision, Recall and F1 for each class individually. For binary problems both classes are always shown here." size="sm" />
                                                                        </div>
                                                                        <div className="grid grid-cols-2 gap-2">
                                                                            {classes.map((cls, clsIdx) => {
                                                                                const btp = matrix[clsIdx]?.[clsIdx] ?? 0;
                                                                                const bfp = matrix.reduce((s, row, ri) => ri !== clsIdx ? s + (row[clsIdx] ?? 0) : s, 0);
                                                                                const bfn = (matrix[clsIdx] ?? []).reduce((s, v, ci) => ci !== clsIdx ? s + v : s, 0);
                                                                                const bprec = (btp + bfp) > 0 ? btp / (btp + bfp) : 0;
                                                                                const brec  = (btp + bfn) > 0 ? btp / (btp + bfn) : 0;
                                                                                const bf1   = bprec + brec > 0 ? (2 * bprec * brec) / (bprec + brec) : 0;
                                                                                return (
                                                                                    <div key={String(cls)} className="flex flex-col items-center p-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                                                                                        <span className="text-[10px] font-semibold text-gray-600 dark:text-gray-300 mb-1.5">{String(cls)}</span>
                                                                                        <div className="grid grid-cols-3 gap-1 text-[10px] w-full">
                                                                                            {([{ l: 'Prec', v: bprec }, { l: 'Rec', v: brec }, { l: 'F1', v: bf1 }] as { l: string; v: number }[]).map(({ l, v }) => (
                                                                                                <div key={l} className="flex flex-col items-center bg-white dark:bg-gray-800 rounded py-1">
                                                                                                    <span className="text-gray-400">{l}</span>
                                                                                                    <span className={`font-mono font-semibold ${v >= 0.8 ? 'text-green-500' : v >= 0.6 ? 'text-yellow-500' : 'text-red-500'}`}>{v.toFixed(2)}</span>
                                                                                                </div>
                                                                                            ))}
                                                                                        </div>
                                                                                    </div>
                                                                                );
                                                                            })}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        );
                                                    })()}
                                                </div>

                                                {/* ROC Curve (if available) */}
                                                {splitData.y_proba && (
                                                    <div className="h-[340px] w-full relative group" id={`${splitName}-roc`}>
                                                        <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                           <button 
                                                             onClick={() => void handleDownload(`${splitName}-roc`, `${splitName}_roc_curve`)}
                                                             disabled={downloadingChart === `${splitName}-roc`}
                                                             className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                             title="Download Graph"
                                                           >
                                                              {downloadingChart === `${splitName}-roc` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-roc` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                           </button>
                                                        </div>
                                                        {(() => {
                                                            if (!selectedRocClass) return <div className="text-center text-xs text-gray-400">Select a class</div>;
                                                            const rocData = calculateROC(splitData.y_true, splitData.y_proba!, selectedRocClass);
                                                            if (!rocData) return <div className="text-center text-xs text-gray-400">ROC not available (multiclass or missing proba)</div>;

                                                            // AUC via trapezoid rule
                                                            const auc = rocData.reduce((sum, pt, i) => {
                                                                if (i === 0) return 0;
                                                                const prev = rocData[i - 1]!;
                                                                return sum + Math.abs(pt.fpr - prev.fpr) * (pt.tpr + prev.tpr) / 2;
                                                            }, 0);

                                                            // Operating point at current threshold
                                                            const proba = splitData.y_proba!;
                                                            let operatingPoint: { fpr: number; tpr: number } | null = null;
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const classIndex = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (classIndex !== -1) {
                                                                const scores = proba.values.map(v => v[classIndex] ?? 0);
                                                                const actual = splitData.y_true.map(t => String(t) === selectedRocClass ? 1 : 0);
                                                                const totalPos = actual.filter(a => a === 1).length;
                                                                const totalNeg = actual.length - totalPos;
                                                                if (totalPos > 0 && totalNeg > 0) {
                                                                    let tp = 0, fp = 0;
                                                                    scores.forEach((s, i) => {
                                                                        if (s >= threshold) {
                                                                            if (actual[i] === 1) tp++;
                                                                            else fp++;
                                                                        }
                                                                    });
                                                                    operatingPoint = { fpr: fp / totalNeg, tpr: tp / totalPos };
                                                                }
                                                            }

                                                            return (
                                                                <>
                                                                    <div className="flex items-center justify-center gap-1.5 mb-1">
                                                                        <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">
                                                                            ROC Curve — {selectedRocClass}
                                                                            <span className="ml-2 font-mono text-purple-600 dark:text-purple-400">AUC={auc.toFixed(3)}</span>
                                                                        </h5>
                                                                        <InfoTooltip
                                                                            text={`ROC (Receiver Operating Characteristic) curve for class "${selectedRocClass}" vs all others. AUC=${auc.toFixed(3)}: closer to 1.0 is better; 0.5 = random. The red dot marks where the model operates at threshold t=${threshold.toFixed(2)} — drag the slider to move it along the curve and see the precision/recall trade-off in real time.`}
                                                                            align="center"
                                                                        />
                                                                    </div>
                                                                    {/* TPR / FPR definitions */}
                                                                    <div className="flex items-center justify-center gap-4 text-[10px] text-gray-400 dark:text-gray-500 mb-1">
                                                                        <div className="flex items-center gap-0.5">
                                                                            <span className="font-semibold">TPR</span>
                                                                            <InfoTooltip text="True Positive Rate (Recall / Sensitivity): TP ÷ (TP + FN). Of all actual positives, how many did the model correctly detect? Higher = fewer misses. This is the Y-axis." size="sm" />
                                                                            <span className="ml-0.5 font-mono">= TP / (TP+FN)</span>
                                                                        </div>
                                                                        <span className="text-gray-300 dark:text-gray-600">·</span>
                                                                        <div className="flex items-center gap-0.5">
                                                                            <span className="font-semibold">FPR</span>
                                                                            <InfoTooltip text="False Positive Rate (Fall-out): FP ÷ (FP + TN). Of all actual negatives, how many were incorrectly flagged as positive? Lower = fewer false alarms. This is the X-axis." size="sm" />
                                                                            <span className="ml-0.5 font-mono">= FP / (FP+TN)</span>
                                                                        </div>
                                                                    </div>
                                                                    <ResponsiveContainer width="100%" height="92%">
                                                                        <ComposedChart data={rocData} margin={{ top: 5, right: 20, bottom: 42, left: 40 }}>
                                                                            <defs>
                                                                                <linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1">
                                                                                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.25} />
                                                                                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.03} />
                                                                                </linearGradient>
                                                                            </defs>
                                                                            <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                            <XAxis
                                                                                type="number"
                                                                                dataKey="fpr"
                                                                                domain={[0, 1]}
                                                                                tickFormatter={(v: number) => v.toFixed(1)}
                                                                                tick={{ fontSize: 10 }}
                                                                                label={{ value: 'FPR (Fall-out)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }}
                                                                            />
                                                                            <YAxis
                                                                                type="number"
                                                                                dataKey="tpr"
                                                                                domain={[0, 1]}
                                                                                tickFormatter={(v: number) => v.toFixed(1)}
                                                                                tick={{ fontSize: 10 }}
                                                                                label={{ value: 'TPR (Recall)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#9ca3af' }}
                                                                            />
                                                                            <Tooltip
                                                                                content={({ active, payload }) => {
                                                                                    if (!active || !payload?.length) return null;
                                                                                    const d = payload[0]?.payload as { fpr: number; tpr: number };
                                                                                    return (
                                                                                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2 text-xs space-y-0.5">
                                                                                            <p className="font-semibold text-gray-600 dark:text-gray-300 mb-1">ROC point</p>
                                                                                            <p className="text-gray-700 dark:text-gray-200">TPR (Recall) <span className="font-mono text-purple-600 dark:text-purple-400">{d.tpr.toFixed(3)}</span></p>
                                                                                            <p className="text-gray-500 dark:text-gray-400">FPR (Fall-out) <span className="font-mono">{d.fpr.toFixed(3)}</span></p>
                                                                                            <p className="text-gray-400 dark:text-gray-500 text-[10px] pt-0.5 border-t border-gray-100 dark:border-gray-700">Precision = TP / (TP+FP) &nbsp;·&nbsp; Recall = TP / (TP+FN)</p>
                                                                                        </div>
                                                                                    );
                                                                                }}
                                                                            />
                                                                            {/* Gradient fill — AUC area */}
                                                                            <Area type="monotone" dataKey="tpr" stroke="none" fill="url(#aucFill)" isAnimationActive={false} legendType="none" />
                                                                            {/* ROC curve */}
                                                                            <Line type="monotone" dataKey="tpr" stroke="#8b5cf6" dot={false} strokeWidth={2.5} name="ROC" isAnimationActive={false} />
                                                                            {/* Random classifier diagonal */}
                                                                            <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="#d1d5db" strokeDasharray="4 3" dot={false} legendType="none" strokeWidth={1} isAnimationActive={false} />
                                                                            {/* Operating point at current threshold */}
                                                                            {operatingPoint && (
                                                                                <Line
                                                                                    data={[operatingPoint]}
                                                                                    dataKey="tpr"
                                                                                    stroke="#ef4444"
                                                                                    dot={{ r: 7, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
                                                                                    activeDot={{ r: 9 }}
                                                                                    isAnimationActive={false}
                                                                                    legendType="none"
                                                                                    name={`t=${threshold.toFixed(2)}`}
                                                                                />
                                                                            )}
                                                                        </ComposedChart>
                                                                    </ResponsiveContainer>
                                                                </>
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
                        )}
                        {/* Per-class comparison — multiclass only; binary shows both classes inline in the overall card */}
                        {evaluationData.problem_type === 'classification' && cmView === 'per-class' && (evaluationData.splits.train?.y_proba?.classes.length ?? 0) > 2 && (() => {
                            // Compute confusion matrix for a split with OvR threshold applied
                            const getMatrix = (splitData: EvaluationSplit) => {
                                const proba = splitData.y_proba;
                                let yTrue: (string | number)[] = splitData.y_true;
                                let yPred: (string | number)[] = splitData.y_pred;
                                if (proba?.labels && proba.labels.length === proba.classes.length) {
                                    const lm = new Map<string, string | number>();
                                    proba.labels.forEach((l, i) => { const c = proba.classes[i]; if (c !== undefined) lm.set(String(l), c); });
                                    yTrue = yTrue.map(y => lm.get(String(y)) ?? y);
                                    yPred = yPred.map(y => lm.get(String(y)) ?? y);
                                }
                                if (proba && selectedRocClass) {
                                    const ll = proba.labels?.length === proba.classes.length ? proba.labels : undefined;
                                    const posIdx = (ll ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                    if (posIdx !== -1) {
                                        const posVal = proba.classes[posIdx];
                                        const orig = [...yPred];
                                        if (posVal !== undefined) {
                                            yPred = proba.values.map((v, i) => {
                                                if ((v[posIdx] ?? 0) >= threshold) return posVal;
                                                let bi = -1, bp = -Infinity;
                                                v.forEach((p, idx) => { if (idx !== posIdx && p > bp) { bp = p; bi = idx; } });
                                                return bi >= 0 ? (proba.classes[bi] ?? orig[i]!) : (orig[i]!);
                                            });
                                        }
                                    }
                                }
                                return calculateConfusionMatrix(yTrue, yPred, proba?.classes);
                            };

                            const renderSplitPerClass = (splitName: string, splitData: EvaluationSplit) => {
                                const { classes, matrix } = getMatrix(splitData);
                                const splitId = `per-class-${splitName}`;
                                return (
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-1.5 mb-1">
                                            <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-300 capitalize">{splitName} Set</h4>
                                            <button
                                                id={`${splitId}-dl`}
                                                onClick={() => void handleDownload(splitId, `${splitName}_per_class`)}
                                                disabled={downloadingChart === splitId}
                                                className="p-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-400 hover:text-blue-600 disabled:opacity-50"
                                                title="Download Per-Class View"
                                            >
                                                {downloadingChart === splitId ? <Loader2 className="w-3 h-3 animate-spin" /> : doneChart === splitId ? <Check className="w-3 h-3 text-green-500" /> : <Download className="w-3 h-3" />}
                                            </button>
                                        </div>
                                        <div id={splitId} className={`grid ${classes.length <= 4 ? 'grid-cols-2' : 'grid-cols-3'} gap-2`}>
                                            {classes.map((cls, clsIdx) => {
                                                const tp = matrix[clsIdx]?.[clsIdx] ?? 0;
                                                const fp = matrix.reduce((s, row, ri) => ri !== clsIdx ? s + (row[clsIdx] ?? 0) : s, 0);
                                                const fn = (matrix[clsIdx] ?? []).reduce((s, v, ci) => ci !== clsIdx ? s + v : s, 0);
                                                const total = matrix.flat().reduce((a, b) => a + b, 0);
                                                const tn = total - tp - fp - fn;
                                                const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
                                                const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
                                                const f1c = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
                                                const isHighlighted = String(cls) === selectedRocClass;
                                                const otherClasses = classes.filter((_, i) => i !== clsIdx);
                                                // Binary: use actual other class name; multiclass: 'Others'
                                                const otherLabel = otherClasses.length === 1 ? String(otherClasses[0]) : 'Others';
                                                const rowLabels = [String(cls), otherLabel];
                                                const cellLbls = [['TP', 'FN'], ['FP', 'TN']];
                                                const recPct = Math.round(rec * 100);
                                                const fnCount = tp + fn > 0 ? fn : 0;
                                                const insight =
                                                    f1c >= 0.8 ? `t=${threshold.toFixed(2)}: catches ${recPct}% of ${String(cls)} — strong.`
                                                    : f1c >= 0.6 ? `t=${threshold.toFixed(2)}: catches ${recPct}% of ${String(cls)} — room to improve.`
                                                    : `t=${threshold.toFixed(2)}: only ${recPct}% caught — model struggles here.`;
                                                const insightTip = [
                                                    `Out of ${tp + fnCount} actual "${String(cls)}" samples, the model correctly caught ${tp} of them = ${recPct}%.`,
                                                    ``,
                                                    `How: Recall = TP ÷ (TP + FN) = ${tp} ÷ ${tp + fnCount} = ${recPct}%`,
                                                    `  TP ${tp}: labelled "${String(cls)}" and actually "${String(cls)}" ✓`,
                                                    `  FN ${fnCount}: actually "${String(cls)}" but predicted as something else ✗`,
                                                    ``,
                                                    `Threshold (≥ ${threshold.toFixed(2)}): the model predicts "${String(cls)}" only when it is ${Math.round(threshold * 100)}%+ confident.`,
                                                    `↑ Raise threshold → harder to trigger, fewer catches (recall ↓), more precise (precision ↑).`,
                                                    `↓ Lower threshold → easier to trigger, more catches (recall ↑), more false alarms (precision ↓).`,
                                                ].join('\n');
                                                const insightColor = f1c >= 0.8 ? 'text-green-600 dark:text-green-400' : f1c >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-500 dark:text-red-400';
                                                return (
                                                    <div key={String(cls)} className={`flex flex-col items-center p-2 rounded-lg border ${isHighlighted ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700'}`}>
                                                        <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5 w-full text-center truncate" title={`${String(cls)} vs Rest`}>{String(cls)} vs Rest</span>
                                                        <div className="flex flex-col">
                                                            <div className="flex mb-0.5 gap-0.5" style={{ marginLeft: '60px' }}>
                                                                <span className="w-14 text-center text-[9px] text-gray-500 dark:text-gray-400 truncate font-medium" title={String(cls)}>{String(cls)}</span>
                                                                <span className="w-14 text-center text-[9px] text-gray-500 dark:text-gray-400 truncate font-medium" title={otherClasses.map(String).join(', ')}>{otherLabel}</span>
                                                            </div>
                                                            {[[tp, fn], [fp, tn]].map((row2, ri) => (
                                                                <div key={ri} className="flex items-center gap-0.5 mb-0.5">
                                                                    <span className="text-right text-[9px] text-gray-500 dark:text-gray-400 pr-1 truncate font-medium" style={{ width: '60px' }} title={rowLabels[ri]}>{rowLabels[ri]}</span>
                                                                    {row2.map((count, ci) => {
                                                                        const isCorrect = ri === ci;
                                                                        const rowMax = Math.max(...row2, 1);
                                                                        const bg = isCorrect
                                                                            ? `rgba(34,197,94,${Math.min((count / rowMax) * 0.75 + 0.1, 0.85)})`
                                                                            : `rgba(239,68,68,${Math.min((count / rowMax) * 0.65 + 0.05, 0.75)})`;
                                                                        return (
                                                                            <div key={ci} className="w-14 h-11 flex flex-col items-center justify-center rounded border border-gray-100 dark:border-gray-700 cursor-default" style={{ backgroundColor: bg }} title={`${cellLbls[ri]?.[ci] ?? ''}=${count}`}>
                                                                                <span className="text-[11px] font-mono font-bold leading-none">{count}</span>
                                                                                <span className="text-[9px] font-semibold opacity-80 mt-0.5">{cellLbls[ri]?.[ci]}</span>
                                                                            </div>
                                                                        );
                                                                    })}
                                                                </div>
                                                            ))}
                                                        </div>
                                                        <div className="mt-1.5 grid grid-cols-3 gap-1 text-[10px] w-full">
                                                            {([{ l: 'Prec', v: prec }, { l: 'Rec', v: rec }, { l: 'F1', v: f1c }] as { l: string; v: number }[]).map(({ l, v }) => (
                                                                <div key={l} className="flex flex-col items-center bg-gray-50 dark:bg-gray-900 rounded py-1">
                                                                    <span className="text-gray-400">{l}</span>
                                                                    <span className={`font-mono font-semibold ${v >= 0.8 ? 'text-green-500' : v >= 0.6 ? 'text-yellow-500' : 'text-red-500'}`}>{v.toFixed(2)}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                        {/* 1-sentence plain-language verdict with hover explanation of the calculation */}
                                                        <div className={`mt-1.5 flex items-center justify-center gap-0.5 ${insightColor}`}>
                                                            <p className="text-[9px] leading-snug text-center">{insight}</p>
                                                            <InfoTooltip text={insightTip} size="sm" />
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                );
                            };

                            const allSplitEntries = Object.entries(evaluationData.splits) as [string, EvaluationSplit][];
                            const trainEntry = showTrainMetrics ? allSplitEntries.find(([n]) => n === 'train') : undefined;
                            const testEntry  = showTestMetrics  ? allSplitEntries.find(([n]) => n === 'test')  : undefined;
                            const valEntry   = showValMetrics   ? allSplitEntries.find(([n]) => n === 'validation') : undefined;

                            return (
                                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        {trainEntry && renderSplitPerClass(trainEntry[0], trainEntry[1])}
                                        {testEntry  && renderSplitPerClass(testEntry[0],  testEntry[1])}
                                        {!trainEntry && !testEntry && (
                                            <p className="col-span-2 text-xs text-gray-400 text-center py-8">Enable Train or Test splits above to compare.</p>
                                        )}
                                    </div>
                                    {valEntry && (
                                        <div className="mt-6 pt-4 border-t border-gray-100 dark:border-gray-700">
                                            {renderSplitPerClass(valEntry[0], valEntry[1])}
                                        </div>
                                    )}
                                </div>
                            );
                        })()}
                        </div>
                    )}
                </div>
              )}

              {/* Pipeline Diff View (L5) */}
              {activeView === 'diff' && (
                <div className="animate-in fade-in duration-300">
                  <PipelineDiffView jobs={selectedJobs} />
                </div>
              )}

              {/* Feature Importance View */}
              {activeView === 'importance' && hasFeatureImportances && (
                <div className="space-y-6 animate-in fade-in duration-300">
                  <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Feature Importance Comparison</h3>
                  {(() => {
                    // Collect all unique features across selected jobs
                    const allFeatures = new Set<string>();
                    featureImportancesByJob.forEach(j => {
                      if (j.importances) Object.keys(j.importances).forEach(f => allFeatures.add(f));
                    });

                    // Build chart data: each feature as a row, each job as a bar
                    const jobsWithData = featureImportancesByJob.filter(j => j.importances !== null);
                    if (jobsWithData.length === 0) return null;

                    // Rank features by average importance (descending), take top 15
                    const featureAvg = Array.from(allFeatures).map(f => {
                      let sum = 0;
                      let count = 0;
                      jobsWithData.forEach(j => {
                        const val = j.importances?.[f];
                        if (val !== undefined) { sum += val; count++; }
                      });
                      return { feature: f, avg: count > 0 ? sum / count : 0 };
                    });
                    featureAvg.sort((a, b) => b.avg - a.avg);
                    const topFeatures = featureAvg.slice(0, 15).map(f => f.feature);

                    const chartData = topFeatures.map(feature => {
                      const row: Record<string, string | number> = { feature };
                      jobsWithData.forEach(j => {
                        const shortId = j.jobId.slice(0, 8);
                        const label = j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
                        row[label] = j.importances?.[feature] ?? 0;
                      });
                      return row;
                    });

                    const barColors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];
                    const barKeys = jobsWithData.map((j, _i) => {
                      const shortId = j.jobId.slice(0, 8);
                      return j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
                    });

                    return (
                      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id="feature-importance-chart">
                        <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                          <button
                            onClick={() => void handleDownload('feature-importance-chart', 'feature_importance_comparison')}
                            disabled={downloadingChart === 'feature-importance-chart'}
                            className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                            title="Download Graph"
                          >
                            {downloadingChart === 'feature-importance-chart' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === 'feature-importance-chart' ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                          </button>
                        </div>
                        <div className="h-[500px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, bottom: 5, left: 120 }}>
                              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                              <XAxis type="number" tick={{ fontSize: 12 }} />
                              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
                              <Tooltip
                                contentStyle={{ backgroundColor: 'rgba(0,0,0,0.85)', border: 'none', borderRadius: '8px', color: '#fff' }}
                                formatter={(value: number) => value.toFixed(4)}
                              />
                              <Legend />
                              {barKeys.map((key, i) => (
                                <Bar key={key} dataKey={key} fill={barColors[i % barColors.length]} radius={[0, 4, 4, 0]} />
                              ))}
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        {topFeatures.length < allFeatures.size && (
                          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">
                            Showing top {topFeatures.length} of {allFeatures.size} features
                          </p>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
