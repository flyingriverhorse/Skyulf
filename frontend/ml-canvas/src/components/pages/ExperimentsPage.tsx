import React, { useState, useEffect, useMemo } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { Filter } from 'lucide-react';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';
import { toPng } from 'html-to-image';
import { deploymentApi } from '../../core/api/deployment';
import { apiClient } from '../../core/api/client';
import { PipelineDiffView } from './experiments/PipelineDiffView';
import type { EvaluationData } from './ExperimentsPage/types';
import { getJobScoringMetric, shortRunId } from './ExperimentsPage/utils/jobMeta';
import { findBestF1Threshold } from './ExperimentsPage/utils/classificationCharts';
import { ComparisonTableView } from './ExperimentsPage/components/ComparisonTableView';
import { FeatureImportanceView } from './ExperimentsPage/components/FeatureImportanceView';
import { BranchComparisonCard } from './ExperimentsPage/components/BranchComparisonCard';
import { MetricsComparisonChart } from './ExperimentsPage/components/MetricsComparisonChart';
import { JobListSidebar } from './ExperimentsPage/components/JobListSidebar';
import { EvaluationView } from './ExperimentsPage/components/EvaluationView';
import { ExperimentsHeader, ViewTabs, type ExperimentsView } from './ExperimentsPage/components/HeaderAndTabs';

// Local helper: split a metric key into split-prefix and base name.
const parseMetricKey = (key: string) => {
  if (key === 'best_score') return { type: 'val', base: 'best_score' };
  if (key.startsWith('train_')) return { type: 'train', base: key.replace('train_', '') };
  if (key.startsWith('test_')) return { type: 'test', base: key.replace('test_', '') };
  if (key.startsWith('val_')) return { type: 'val', base: key.replace('val_', '') };
  return { type: 'other', base: key };
};

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

  // Table expansion states (lifted so they survive view switches)
  const [isMetricsExpanded, setIsMetricsExpanded] = useState(true);
  const [isParamsExpanded, setIsParamsExpanded] = useState(true);
  const [isTuningExpanded, setIsTuningExpanded] = useState(true);
  const [isPipelineExpanded, setIsPipelineExpanded] = useState(true);

  // View state
  const [activeView, setActiveView] = useState<ExperimentsView>('charts');
  const [evaluationData, setEvaluationData] = useState<EvaluationData | null>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalJobId, setEvalJobId] = useState<string | null>(null);
  const [downloadingChart, setDownloadingChart] = useState<string | null>(null);
  const [doneChart, setDoneChart] = useState<string | null>(null);
  const [selectedRocClass, setSelectedRocClass] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [cmView, setCmView] = useState<'overall' | 'per-class'>('overall');
  const [selectedRegressionSplit, setSelectedRegressionSplit] = useState<string | null>(null);

  // Best F1 threshold — recomputed only when class or evaluation data changes, not on every slider drag
  const bestF1Info = useMemo(() => {
    if (!evaluationData || !selectedRocClass) return null;
    const splits = evaluationData.splits;
    const refSplit = splits.val ?? splits.test ?? splits.train;
    const splitLabel = splits.val ? 'val' : splits.test ? 'test' : 'train';
    if (!refSplit?.y_proba) return null;
    const result = findBestF1Threshold(refSplit.y_true, refSplit.y_proba, selectedRocClass);
    if (!result) return null;
    const evalJob = evalJobId ? jobs.find(j => j.job_id === evalJobId) : null;
    const metricName = (evalJob ? getJobScoringMetric(evalJob) : undefined) ?? 'f1';
    return { ...result, splitLabel, metricName };
  }, [evaluationData, selectedRocClass, evalJobId, jobs]);

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
      console.error('Failed to fetch datasets', e);
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
    // Stale-while-revalidate: keep showing the previously rendered
    // charts while the new run loads. Setting `evaluationData` to
    // null here would unmount the entire panel and flash the
    // spinner on every job switch \u2014 the "blink" the user reported
    // when clicking between runs in the Model Evaluation tab.
    setIsEvalLoading(true);
    setEvalError(null);
    setEvalJobId(jobId);
    try {
      const res = await apiClient.get(`/pipeline/jobs/${jobId}/evaluation`);
      setEvaluationData(res.data);
    } catch (err: unknown) {
      console.error('Failed to fetch evaluation data', err);
      setEvalError((err as { response?: { data?: { detail?: string } } }).response?.data?.detail || 'Failed to fetch evaluation data');
      setEvaluationData(null);
    } finally {
      setIsEvalLoading(false);
    }
  };

  // Effect to fetch evaluation data when view changes or selection changes
  useEffect(() => {
    if (activeView === 'evaluation') {
      if (!evalJobId && selectedJobIds.length > 0) {
        void fetchEvaluationData(selectedJobIds[0]!);
      } else if (evalJobId && !selectedJobIds.includes(evalJobId) && selectedJobIds.length > 0) {
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
    const metrics = job.metrics || job.result?.metrics || {};
    return { name: shortRunId(job), ...metrics };
  });

  // Get all unique metric keys from selected jobs (numeric only, filtered by visibility)
  const metricKeys = Array.from(new Set(
    selectedJobs.flatMap(job => {
      const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
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

  // Group keys by base metric name for the metric-tab selector
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
      <ExperimentsHeader
        datasets={datasets}
        selectedDatasetId={selectedDatasetId}
        setSelectedDatasetId={setSelectedDatasetId}
        filterType={filterType}
        setFilterType={setFilterType}
      />

      <div className="flex-1 flex overflow-hidden">
        <JobListSidebar
          filteredJobs={filteredJobs}
          selectedJobIds={selectedJobIds}
          isSidebarCollapsed={isSidebarCollapsed}
          setIsSidebarCollapsed={setIsSidebarCollapsed}
          toggleJobSelection={toggleJobSelection}
          hasMore={hasMore}
          isLoading={isLoading}
          loadMoreJobs={loadMoreJobs}
          handlePromote={handlePromote}
          handleDeploy={handleDeploy}
          getDuration={getDuration}
        />

        {/* Comparison Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {selectedJobs.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <Filter className="w-12 h-12 mb-4 opacity-20" />
              <p>Select runs from the sidebar to compare them.</p>
            </div>
          ) : (
            <div className="space-y-6">
              <ViewTabs
                activeView={activeView}
                setActiveView={setActiveView}
                hasFeatureImportances={hasFeatureImportances}
              />

              <BranchComparisonCard selectedJobs={selectedJobs} getDuration={getDuration} />

              {activeView === 'charts' && (
                <MetricsComparisonChart
                  metricsData={metricsData}
                  metricGroups={metricGroups}
                  availableMetrics={availableMetrics}
                  activeMetric={activeMetric}
                  setSelectedMetric={setSelectedMetric}
                  showTrainMetrics={showTrainMetrics}
                  setShowTrainMetrics={setShowTrainMetrics}
                  showTestMetrics={showTestMetrics}
                  setShowTestMetrics={setShowTestMetrics}
                  showValMetrics={showValMetrics}
                  setShowValMetrics={setShowValMetrics}
                  showCvMetrics={showCvMetrics}
                  setShowCvMetrics={setShowCvMetrics}
                />
              )}

              {activeView === 'table' && (
                <ComparisonTableView
                  selectedJobs={selectedJobs}
                  metricKeys={metricKeys}
                  isPipelineExpanded={isPipelineExpanded}
                  setIsPipelineExpanded={setIsPipelineExpanded}
                  isMetricsExpanded={isMetricsExpanded}
                  setIsMetricsExpanded={setIsMetricsExpanded}
                  isParamsExpanded={isParamsExpanded}
                  setIsParamsExpanded={setIsParamsExpanded}
                  isTuningExpanded={isTuningExpanded}
                  setIsTuningExpanded={setIsTuningExpanded}
                />
              )}

              {activeView === 'evaluation' && (
                <EvaluationView
                  selectedJobIds={selectedJobIds}
                  evalJobId={evalJobId}
                  fetchEvaluationData={fetchEvaluationData}
                  isEvalLoading={isEvalLoading}
                  evalError={evalError}
                  evaluationData={evaluationData}
                  selectedRegressionSplit={selectedRegressionSplit}
                  setSelectedRegressionSplit={setSelectedRegressionSplit}
                  showTrainMetrics={showTrainMetrics}
                  setShowTrainMetrics={setShowTrainMetrics}
                  showTestMetrics={showTestMetrics}
                  setShowTestMetrics={setShowTestMetrics}
                  showValMetrics={showValMetrics}
                  setShowValMetrics={setShowValMetrics}
                  threshold={threshold}
                  setThreshold={setThreshold}
                  selectedRocClass={selectedRocClass}
                  setSelectedRocClass={setSelectedRocClass}
                  cmView={cmView}
                  setCmView={setCmView}
                  bestF1Info={bestF1Info}
                  handleDownload={handleDownload}
                  downloadingChart={downloadingChart}
                  doneChart={doneChart}
                />
              )}

              {activeView === 'diff' && (
                <PipelineDiffView jobs={selectedJobs} />
              )}

              {activeView === 'importance' && hasFeatureImportances && (
                <FeatureImportanceView
                  featureImportancesByJob={featureImportancesByJob}
                  handleDownload={handleDownload}
                  downloadingChart={downloadingChart}
                  doneChart={doneChart}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
