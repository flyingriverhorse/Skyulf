import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { Filter } from 'lucide-react';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';
import { toPng } from 'html-to-image';
import { deploymentApi } from '../../core/api/deployment';
import { apiClient } from '../../core/api/client';
import { formatDuration } from '../../core/utils/format';
import { PipelineDiffView } from './experiments/PipelineDiffView';
import type { EvaluationData, ShapExplanationData } from './ExperimentsPage/types';
import { getJobScoringMetric, getTaskForModelType, mapJobMetricToDropdown, shortRunId, type ThresholdMetric } from './ExperimentsPage/utils/jobMeta';
import { registryApi, type RegistryItem } from '../../core/api/registry';
import { findBestThreshold } from './ExperimentsPage/utils/classificationCharts';
import { ComparisonTableView } from './ExperimentsPage/components/ComparisonTableView';
import { FeatureImportanceView } from './ExperimentsPage/components/FeatureImportanceView';
import { ShapExplainabilityView } from './ExperimentsPage/components/ShapExplainabilityView';
import { BranchComparisonCard } from './ExperimentsPage/components/BranchComparisonCard';
import { MetricsComparisonChart } from './ExperimentsPage/components/MetricsComparisonChart';
import { JobListSidebar } from './ExperimentsPage/components/JobListSidebar';
import { EvaluationView } from './ExperimentsPage/components/EvaluationView';
import { SegmentationView } from './ExperimentsPage/components/SegmentationView';
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
  // Ref mirror of `jobs`, read (not depended on) inside fetchEvaluationData
  // below so looking up the job's own scoring metric doesn't force that
  // callback to be treated as reactive on every jobs-list refresh/poll.
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [filterType, setFilterType] = useState<'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation'>('all');
  const [datasets, setDatasets] = useState<{id: string, name: string}[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('all');
  const [registryItems, setRegistryItems] = useState<RegistryItem[]>([]);

  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Metric visibility toggles
  const [showTrainMetrics, setShowTrainMetrics] = useState(true);
  const [showTestMetrics, setShowTestMetrics] = useState(true);
  const [showValMetrics, setShowValMetrics] = useState(true);
  // Cross-Validation defaults to unchecked/hidden in the visual chart
  // comparison — it's a single cross-validated score, not comparable
  // split-for-split against train/test/val bars, so it would otherwise
  // clutter the default view. Still toggleable by the user via its
  // checkbox when they actually want to see it.
  const [showCvMetrics, setShowCvMetrics] = useState(false);

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
  // Which metric the classification best-threshold scan optimizes for.
  // Reset per-job in fetchEvaluationData below, defaulting to the job's
  // own scoring metric (via mapJobMetricToDropdown) instead of always F1.
  const [selectedThresholdMetric, setSelectedThresholdMetric] = useState<ThresholdMetric>('f1_weighted');

  // Best-threshold badge(s) — recomputed only when class, metric, visible
  // splits, or evaluation data changes, not on every slider drag.
  //
  // Computes one badge PER currently-visible split (per the "Splits:"
  // toggles) rather than a single winner, so the user can see and apply
  // the best threshold for Train, Test, and Validation independently
  // instead of one hidden-split value being silently applied regardless
  // of what's on screen. Falls back to showing every split with proba
  // data if none are currently visible (e.g. all toggles off).
  //
  // Note: the splits dict key is `'validation'`, not `'val'` — mirrors the
  // same key the regression-tab fix above already corrected.
  const bestMetricInfos = useMemo(() => {
    if (!evaluationData || !selectedRocClass || evaluationData.problem_type === 'clustering') return [];
    const splits = evaluationData.splits;
    const priority: Array<{ key: string; visible: boolean; label: string }> = [
      { key: 'validation', visible: showValMetrics, label: 'validation' },
      { key: 'test', visible: showTestMetrics, label: 'test' },
      { key: 'train', visible: showTrainMetrics, label: 'train' },
    ];
    const candidates = priority.filter(p => splits[p.key]?.y_proba);
    const visibleCandidates = candidates.filter(p => p.visible);
    const toCompute = visibleCandidates.length > 0 ? visibleCandidates : candidates;
    return toCompute.flatMap(({ key, label }) => {
      const refSplit = splits[key];
      if (!refSplit?.y_proba) return [];
      const result = findBestThreshold(refSplit.y_true, refSplit.y_proba, selectedRocClass, selectedThresholdMetric);
      if (!result) return [];
      return [{ ...result, splitLabel: label, metricName: selectedThresholdMetric }];
    });
  }, [evaluationData, selectedRocClass, selectedThresholdMetric, showValMetrics, showTestMetrics, showTrainMetrics]);


  useEffect(() => {
    fetchJobs();
    void fetchDatasets();
    void fetchRegistryItems();
  }, [fetchJobs]);

  useEffect(() => {
    if (evaluationData?.problem_type !== 'clustering' && evaluationData?.splits.train?.y_proba?.classes && evaluationData.splits.train.y_proba.classes.length > 0) {
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
      toast.error('Failed to load datasets', 'The dataset filter may be incomplete. Please retry.');
    }
  };

  // One-time fetch of the node registry so job task types (Classification /
  // Regression / Text Classification / Segmentation) can be derived from
  // each job's model_type, mirroring TrainingSettings.tsx's tag-based model
  // filtering (plan §0.5/§0.6).
  const fetchRegistryItems = async () => {
    try {
      const nodes = await registryApi.getAllNodes();
      setRegistryItems(nodes);
    } catch (e) {
      console.error('Failed to fetch node registry', e);
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
    // Default the metric dropdown to this job's own scoring metric (not
    // always F1) — the user can still change it afterward for this job.
    const job = jobsRef.current.find(j => j.job_id === jobId);
    setSelectedThresholdMetric(mapJobMetricToDropdown(job ? getJobScoringMetric(job) : undefined));
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
    if (activeView === 'evaluation' || activeView === 'segmentation') {
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

  const filteredJobs = useMemo(() => jobs.filter(job => {
    const typeMatch = filterType === 'all' || getTaskForModelType(job.model_type, registryItems) === filterType;
    const datasetMatch = selectedDatasetId === 'all' || job.dataset_id === selectedDatasetId;
    const statusMatch = job.status === 'completed';
    return typeMatch && datasetMatch && statusMatch;
  }).sort((a, b) => {
    // Promoted jobs float to top
    if (a.promoted_at && !b.promoted_at) return -1;
    if (!a.promoted_at && b.promoted_at) return 1;
    return 0;
  }), [jobs, filterType, selectedDatasetId, registryItems]);

  const selectedJobs = useMemo(
    () => jobs.filter(job => selectedJobIds.includes(job.job_id)),
    [jobs, selectedJobIds]
  );

  const toggleJobSelection = (jobId: string) => {
    setSelectedJobIds(prev =>
      prev.includes(jobId)
        ? prev.filter(id => id !== jobId)
        : [...prev, jobId]
    );
  };

  // Prepare data for charts
  const metricsData = useMemo(() => selectedJobs.map(job => {
    const metrics = job.metrics || job.result?.metrics || {};
    return { name: shortRunId(job), ...metrics };
  }), [selectedJobs]);

  // Unfiltered union of numeric metric keys across selected jobs — used to
  // decide which of the Train/Test/Validation/CV checkboxes are even worth
  // showing (a checkbox for a split none of the selected jobs have is dead
  // UI that toggles nothing).
  const rawMetricKeys = useMemo(() => Array.from(new Set(
    selectedJobs.flatMap(job => {
      const m = (job.metrics || job.result?.metrics || {}) as Record<string, unknown>;
      return Object.keys(m).filter(k => {
        const val = m[k];
        return typeof val === 'number' && !isNaN(val);
      });
    })
  )), [selectedJobs]);

  const hasTrainMetrics = useMemo(() => rawMetricKeys.some(k => k.startsWith('train_')), [rawMetricKeys]);
  const hasTestMetrics = useMemo(() => rawMetricKeys.some(k => k.startsWith('test_')), [rawMetricKeys]);
  const hasValMetrics = useMemo(() => rawMetricKeys.some(k => k.startsWith('val_')), [rawMetricKeys]);
  const hasCvMetrics = useMemo(() => rawMetricKeys.some(k => k.startsWith('cv_') || k === 'best_score'), [rawMetricKeys]);

  // Get all unique metric keys from selected jobs (numeric only, filtered by visibility)
  const metricKeys = useMemo(() => Array.from(new Set(
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
  }), [selectedJobs, showTrainMetrics, showTestMetrics, showValMetrics, showCvMetrics]);

  // Group keys by base metric name for the metric-tab selector
  const metricGroups = useMemo(() => {
    const groups = new Map<string, string[]>();
    metricKeys.forEach(key => {
      const { base } = parseMetricKey(key);
      if (!groups.has(base)) groups.set(base, []);
      groups.get(base)?.push(key);
    });
    groups.forEach((keys, base) => {
      groups.set(base, keys.sort());
    });
    return groups;
  }, [metricKeys]);

  const availableMetrics = useMemo(() => Array.from(metricGroups.keys()).sort(), [metricGroups]);
  const activeMetric = (selectedMetric && availableMetrics.includes(selectedMetric))
    ? selectedMetric
    : availableMetrics[0] || null;

  // Feature Importances across selected jobs
  const featureImportancesByJob = useMemo(() => selectedJobs.map(job => {
    const result = (job.result ?? {}) as Record<string, unknown>;
    const metrics = result.metrics as Record<string, unknown> | undefined;
    const raw = (metrics?.feature_importances ?? result.feature_importances) as Record<string, number> | undefined;
    return { jobId: job.job_id, modelType: job.model_type ?? 'unknown', importances: raw ?? null };
  }), [selectedJobs]);
  const hasFeatureImportances = useMemo(
    () => featureImportancesByJob.some(j => j.importances !== null),
    [featureImportancesByJob]
  );

  // SHAP explanations across selected jobs (summary + per-sample data)
  const shapExplanationByJob = useMemo(() => selectedJobs.map(job => {
    const result = (job.result ?? {}) as Record<string, unknown>;
    const metrics = result.metrics as Record<string, unknown> | undefined;
    const raw = (metrics?.shap_explanation ?? result.shap_explanation) as ShapExplanationData | undefined;
    return { jobId: job.job_id, modelType: job.model_type ?? 'unknown', shapExplanation: raw ?? null };
  }), [selectedJobs]);
  const hasShapSummary = useMemo(
    () => shapExplanationByJob.some(j => j.shapExplanation !== null),
    [shapExplanationByJob]
  );

  // Segmentation (clustering) jobs — detected from job.model_type via the
  // same tag-based task lookup used for the filterType tabs, rather than
  // fetching evaluation data for every selected job.
  const hasSegmentation = useMemo(
    () => selectedJobs.some(j => getTaskForModelType(j.model_type, registryItems) === 'segmentation'),
    [selectedJobs, registryItems]
  );

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
          getDuration={formatDuration}
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
                hasShapSummary={hasShapSummary}
                hasSegmentation={hasSegmentation}
              />

              <BranchComparisonCard selectedJobs={selectedJobs} getDuration={formatDuration} />

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
                  hasTrainMetrics={hasTrainMetrics}
                  hasTestMetrics={hasTestMetrics}
                  hasValMetrics={hasValMetrics}
                  hasCvMetrics={hasCvMetrics}
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
                  selectedMetric={selectedThresholdMetric}
                  setSelectedMetric={setSelectedThresholdMetric}
                  bestMetricInfos={bestMetricInfos}
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

              {activeView === 'shap' && hasShapSummary && (
                <ShapExplainabilityView
                  shapExplanationByJob={shapExplanationByJob}
                  handleDownload={handleDownload}
                  downloadingChart={downloadingChart}
                  doneChart={doneChart}
                />
              )}

              {activeView === 'segmentation' && hasSegmentation && (
                <SegmentationView
                  selectedJobIds={selectedJobIds}
                  evalJobId={evalJobId}
                  fetchEvaluationData={fetchEvaluationData}
                  isEvalLoading={isEvalLoading}
                  evalError={evalError}
                  evaluationData={evaluationData}
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
