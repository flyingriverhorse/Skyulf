import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { X, RefreshCw, CheckCircle, AlertCircle, ArrowLeft, Database, Terminal, Square, FileText, LayoutDashboard, ChevronDown, Zap, CheckCircle2, Search, Filter } from 'lucide-react';
import { JobInfo } from '../../core/api/jobs';
import { useEscapeKey } from '../../core/hooks/useEscapeKey';
import { formatMetricName } from '../../core/utils/format';
import { clickableProps } from '../../core/utils/a11y';
import { useJobPolling, isTerminalStatus } from '../../core/hooks/useJobPolling';
import { StatusBadge } from '../shared/StatusBadge';
import { VirtualList } from '../shared/VirtualList';
import { useConfirm } from '../shared';
import { toast } from '../../core/toast';

const formatMetricValue = (key: string, value: number): string => {
  if (key.endsWith('_std')) return value.toFixed(6);
  return value.toFixed(4);
};

/** Extract the scoring metric name from a job's result or config. */
const getScoringMetric = (job: JobInfo): string | undefined => {
  const result = job.result as Record<string, unknown> | undefined;
  if (result?.scoring_metric) return result.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  return tuning?.metric as string | undefined;
};

/** Renders feature_importances from result.metrics or result directly as a sorted bar table. */
const FeatureImportancesSection: React.FC<{ result: Record<string, unknown> }> = ({ result }) => {
  const metrics = result.metrics as Record<string, unknown> | undefined;
  const raw = (metrics?.feature_importances ?? result.feature_importances) as Record<string, number> | undefined;
  if (!raw || typeof raw !== 'object') return null;

  const sorted = Object.entries(raw).sort(([, a], [, b]) => b - a).slice(0, 5);
  if (sorted.length === 0) return null;
  const maxVal = sorted[0]![1] || 1;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Feature Importances</h4>
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3 max-h-64 overflow-y-auto">
        {sorted.map(([feature, importance]) => (
          <div key={feature} className="flex items-center gap-2 py-1">
            <span className="text-xs text-gray-600 dark:text-gray-300 w-32 truncate shrink-0" title={feature}>{feature}</span>
            <div className="flex-1 h-4 bg-gray-100 dark:bg-gray-700 rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 dark:bg-blue-400 rounded"
                style={{ width: `${(importance / maxVal) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-500 dark:text-gray-400 w-14 text-right shrink-0">{importance.toFixed(4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export const JobsDrawer: React.FC = () => {
  const { 
    isDrawerOpen, 
    toggleDrawer, 
    jobs, 
    isLoading, 
    activeTab, 
    setTab,
    fetchJobs,
    hasMore,
    loadMoreJobs,
    activeParallelRun
  } = useJobStore();

  const [selectedJob, setSelectedJob] = useState<JobInfo | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);

  useEscapeKey(toggleDrawer, isDrawerOpen);

  // Reset to list view whenever the drawer re-opens
  useEffect(() => {
    if (isDrawerOpen) setSelectedJob(null);
  }, [isDrawerOpen]);

  if (!isDrawerOpen) return null;

  const tabJobs = jobs.filter(job => job.job_type === activeTab);
  
  // Derive unique model types and statuses from current tab's jobs
  const modelTypes = [...new Set(tabJobs.map(j => j.model_type).filter(Boolean))] as string[];
  const statuses = [...new Set(tabJobs.map(j => j.status))];

  const filteredJobs = tabJobs.filter(job => {
    if (statusFilter !== 'all' && job.status !== statusFilter) return false;
    if (modelFilter !== 'all' && job.model_type !== modelFilter) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchesId = job.job_id.toLowerCase().includes(q);
      const matchesDataset = (job.dataset_name || job.dataset_id || '').toLowerCase().includes(q);
      const matchesModel = (job.model_type || '').toLowerCase().includes(q);
      if (!matchesId && !matchesDataset && !matchesModel) return false;
    }
    return true;
  });

  return (
    <div className="fixed inset-0 z-50 flex justify-center items-center">
      {/* Backdrop */}
      {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events,jsx-a11y/no-static-element-interactions -- backdrop dismiss zone */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={() => toggleDrawer(false)}
      />
      
      {/* Modal Content */}
      <div className="relative w-[1200px] max-w-[95vw] h-[85vh] bg-white dark:bg-gray-800 shadow-2xl rounded-lg flex flex-col border border-gray-200 dark:border-gray-700 overflow-hidden transition-all">
        
        {selectedJob ? (
            <JobDetailsView job={selectedJob} onBack={() => { setSelectedJob(null); }} onClose={() => toggleDrawer(false)} />
        ) : (
            <>
                {/* Header */}
                <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
                <h2 className="font-semibold text-gray-800 dark:text-gray-100">Job History</h2>
                <div className="flex items-center gap-2">
                    <button 
                    onClick={() => fetchJobs()}
                    className={`p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 ${isLoading ? 'animate-spin' : ''}`}
                    title="Refresh"
                    >
                    <RefreshCw className="w-4 h-4" />
                    </button>
                    <button 
                    onClick={() => toggleDrawer(false)}
                    className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                    >
                    <X className="w-4 h-4" />
                    </button>
                </div>
                </div>

                {/* Parallel Run Progress Banner */}
                {activeParallelRun && (() => {
                  const total = activeParallelRun.jobIds.length;
                  const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
                  const doneCount = activeParallelRun.jobIds.filter(id => {
                    const j = jobs.find(job => job.job_id === id);
                    return j && TERMINAL.has(j.status);
                  }).length;
                  const pct = Math.round((doneCount / total) * 100);
                  const isDone = doneCount === total;
                  return (
                    <div className={`px-4 py-2.5 border-b flex items-center gap-3 ${
                      isDone
                        ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700/50'
                        : 'bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700/50'
                    }`}>
                      {isDone
                        ? <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400 shrink-0" />
                        : <Zap className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                      }
                      <span className={`text-sm font-medium ${
                        isDone
                          ? 'text-green-800 dark:text-green-200'
                          : 'text-amber-800 dark:text-amber-200'
                      }`}>
                        {isDone ? 'All branches complete!' : `Parallel Run: ${doneCount}/${total} branches complete`}
                      </span>
                      <div className={`flex-1 h-2 rounded-full overflow-hidden ${
                        isDone ? 'bg-green-200 dark:bg-green-800' : 'bg-amber-200 dark:bg-amber-800'
                      }`}>
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isDone ? 'bg-green-500 dark:bg-green-400' : 'bg-amber-500 dark:bg-amber-400'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className={`text-xs font-mono shrink-0 ${
                        isDone ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'
                      }`}>{pct}%</span>
                    </div>
                  );
                })()}

                {/* Tabs */}
                <div className="flex border-b border-gray-200 dark:border-gray-700">
                <button
                    className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === 'advanced_tuning' 
                        ? 'border-purple-500 text-purple-600 dark:text-purple-400 bg-purple-50/50 dark:bg-purple-900/20' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => setTab('advanced_tuning')}
                >
                    Advanced Training
                </button>
                <button
                    className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === 'basic_training' 
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-50/50 dark:bg-blue-900/20' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => setTab('basic_training')}
                >
                    Standard Training
                </button>
                </div>

                {/* Filter Bar */}
                <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="relative flex-1">
                      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none z-10" />
                      <input
                        type="text"
                        placeholder="Search by job ID, dataset, or model..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-8 pr-3 py-1.5 text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500 text-gray-700 dark:text-gray-200 placeholder-gray-400"
                      />
                    </div>
                    <button
                      onClick={() => setShowFilters(!showFilters)}
                      className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-md border transition-colors ${
                        showFilters || statusFilter !== 'all' || modelFilter !== 'all'
                          ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400'
                          : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                      }`}
                    >
                      <Filter className="w-3.5 h-3.5" />
                      Filters
                      {(statusFilter !== 'all' || modelFilter !== 'all') && (
                        <span className="w-4 h-4 flex items-center justify-center bg-blue-500 text-white rounded-full text-[10px] font-bold">
                          {(statusFilter !== 'all' ? 1 : 0) + (modelFilter !== 'all' ? 1 : 0)}
                        </span>
                      )}
                    </button>
                  </div>
                  {showFilters && (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</span>
                        <select
                          value={statusFilter}
                          onChange={(e) => setStatusFilter(e.target.value)}
                          className="text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="all">All</option>
                          {statuses.map(s => (
                            <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
                          ))}
                        </select>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model</span>
                        <select
                          value={modelFilter}
                          onChange={(e) => setModelFilter(e.target.value)}
                          className="text-xs bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="all">All</option>
                          {modelTypes.map(m => (
                            <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>
                          ))}
                        </select>
                      </div>
                      {(statusFilter !== 'all' || modelFilter !== 'all') && (
                        <button
                          onClick={() => { setStatusFilter('all'); setModelFilter('all'); }}
                          className="text-[10px] text-blue-500 hover:underline"
                        >
                          Clear all
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* List Header */}
                <div className="grid grid-cols-12 gap-4 px-6 py-2 bg-gray-50 dark:bg-gray-900/50 border-b border-gray-200 dark:border-gray-700 text-xs font-medium text-gray-500 dark:text-gray-400">
                    <div className="col-span-2">Status</div>
                    <div className="col-span-2">Dataset / Model</div>
                    <div className="col-span-3">Job ID</div>
                    <div className="col-span-2">Started</div>
                    <div className="col-span-1">Duration</div>
                    <div className="col-span-2">Score</div>
                </div>

                {/* List — virtualized once length crosses the threshold (#15). */}
                <div className="flex-1 flex flex-col overflow-hidden bg-gray-50/30 dark:bg-gray-900/30">
                {filteredJobs.length === 0 ? (
                    <div className="text-center py-10 text-gray-400 dark:text-gray-500 text-sm">
                    {searchQuery || statusFilter !== 'all' || modelFilter !== 'all'
                      ? 'No jobs match the current filters.'
                      : `No ${activeTab === 'advanced_tuning' ? 'advanced training' : 'training'} jobs found.`
                    }
                    </div>
                ) : (
                    <>
                        <VirtualList
                            items={filteredJobs}
                            getKey={(job) => job.job_id}
                            estimateSize={84}
                            className="flex-1 overflow-y-auto p-4 space-y-2"
                            renderItem={(job) => (
                                <div className="pb-2">
                                    <JobRow job={job} onClick={() => { setSelectedJob(job); }} />
                                </div>
                            )}
                        />
                        
                        {hasMore && (
                            <div className="flex-none flex justify-center pt-2 pb-4">
                                <button 
                                    onClick={() => loadMoreJobs()}
                                    disabled={isLoading}
                                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50 flex items-center gap-1"
                                >
                                    {isLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <ChevronDown className="w-3 h-3" />}
                                    Load More History
                                </button>
                            </div>
                        )}
                    </>
                )}
                </div>
            </>
        )}
      </div>
    </div>
  );
};

const JobDetailsView: React.FC<{ job: JobInfo; onBack: () => void; onClose: () => void }> = ({ job: initialJob, onBack, onClose }) => {
    const { cancelJob } = useJobStore();
    const confirm = useConfirm();
    const [activeTab, setActiveTab] = useState<'overview' | 'logs'>('overview');
    const [isCancelling, setIsCancelling] = useState(false);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Poll the single job until terminal. We feed an empty array once
    // the initial job is already terminal so the hook does no work,
    // and otherwise let it stop itself on the next terminal snapshot.
    const pollIds = useMemo(
        () => (isTerminalStatus(initialJob.status) ? [] : [initialJob.job_id]),
        [initialJob.job_id, initialJob.status],
    );
    const { jobs: polledJobs } = useJobPolling(pollIds, { intervalMs: 2000 });
    const job: JobInfo = polledJobs[initialJob.job_id] ?? initialJob;

    // Auto-scroll logs
    useEffect(() => {
        if (activeTab === 'logs' && logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [job.logs, activeTab]);

    const handleCancel = async () => {
        const ok = await confirm({
            title: 'Stop job?',
            message: 'Are you sure you want to stop this job?',
            confirmLabel: 'Stop',
            variant: 'danger',
        });
        if (!ok) return;
        setIsCancelling(true);
        try {
            await cancelJob(job.job_id);
        } catch (e) {
            toast.error('Failed to cancel job');
        } finally {
            setIsCancelling(false);
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
                <div className="flex items-center gap-3">
                    <button onClick={onBack} className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-500">
                        <ArrowLeft className="w-4 h-4" />
                    </button>
                    <div>
                        <h2 className="font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                            Job Details
                            <span className="text-xs font-normal text-gray-500 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                                {job.job_id.slice(0, 8)}
                            </span>
                        </h2>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {(job.status === 'running' || job.status === 'queued') && (
                        <button 
                            onClick={() => { void handleCancel(); }}
                            disabled={isCancelling}
                            className="flex items-center gap-1 px-3 py-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/40 rounded text-xs font-medium transition-colors border border-red-200 dark:border-red-800"
                        >
                            <Square className="w-3 h-3 fill-current" />
                            {isCancelling ? 'Stopping...' : 'Stop Job'}
                        </button>
                    )}
                    <button onClick={onClose} className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400">
                        <X className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-gray-200 dark:border-gray-700 px-4">
                <button
                    className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                    activeTab === 'overview' 
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => { setActiveTab('overview'); }}
                >
                    <LayoutDashboard className="w-4 h-4" />
                    Overview
                </button>
                <button
                    className={`py-3 px-4 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                    activeTab === 'logs' 
                        ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                        : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                    }`}
                    onClick={() => { setActiveTab('logs'); }}
                >
                    <FileText className="w-4 h-4" />
                    Live Logs
                    {job.status === 'running' && <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />}
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                {activeTab === 'overview' ? (
                    <div className="space-y-6">
                        {/* Status Section */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Status</div>
                                <div className="font-medium capitalize flex items-center gap-2 text-gray-800 dark:text-gray-200">
                                    {job.status}
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Dataset</div>
                                <div className="font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                    <Database className="w-3 h-3 text-gray-400" />
                                    {job.dataset_name || job.dataset_id || 'Unknown'}
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Duration</div>
                                <div className="font-medium text-gray-800 dark:text-gray-200 font-mono">
                                    {job.start_time && job.end_time 
                                        ? `${Math.round((new Date(job.end_time).getTime() - new Date(job.start_time).getTime()) / 1000)}s` 
                                        : '-'}
                                </div>
                            </div>
                        </div>

                        {/* Error Section */}
                        {job.error && (
                            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-100 dark:border-red-900/30 rounded-lg">
                                <h3 className="text-sm font-medium text-red-800 dark:text-red-300 mb-2 flex items-center gap-2">
                                    <AlertCircle className="w-4 h-4" />
                                    Error Log
                                </h3>
                                <pre className="text-xs text-red-700 dark:text-red-400 whitespace-pre-wrap font-mono">
                                    {job.error}
                                </pre>
                            </div>
                        )}

                        {/* Results Section */}
                        {job.result && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-medium text-gray-800 dark:text-gray-200 flex items-center gap-2">
                                        <Terminal className="w-4 h-4" />
                                        Execution Results
                                    </h3>
                                    {job.status === 'completed' && (
                                        <span className="text-xs px-2 py-1 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full flex items-center gap-1 border border-green-200 dark:border-green-800 font-medium">
                                            <CheckCircle className="w-3 h-3" /> Model Ready
                                        </span>
                                    )}
                                </div>
                                
                                {job.job_type === 'basic_training' && !!(job.result as Record<string, unknown>).metrics && (
                                    <div className="space-y-4">
                                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                            {Object.entries((job.result as Record<string, unknown>).metrics as Record<string, unknown>)
                                                .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
                                                .map(([k, v]) => (
                                                <div key={k} className={`p-3 border rounded-lg ${k.startsWith('cv_') ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'}`}>
                                                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 capitalize">{k.replace(/_/g, ' ')}</div>
                                                    <div className={`font-mono font-medium ${k.startsWith('cv_') ? 'text-purple-600 dark:text-purple-400' : 'text-blue-600 dark:text-blue-400'}`}>
                                                        {typeof v === 'number' ? formatMetricValue(k, v) : String(v)}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                        <FeatureImportancesSection result={job.result as Record<string, unknown>} />
                                    </div>
                                )}

                                {job.job_type === 'advanced_tuning' && (
                                    <div className="space-y-4">
                                        {/* Tuning Configuration */}
                                        {job.graph && (
                                            <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-100 dark:border-gray-700">
                                                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">Tuning Configuration</h4>
                                                <div className="grid grid-cols-2 gap-4 text-xs">
                                                    {(() => {
                                                        const node = (job.graph?.nodes as Array<{ node_id: string; params?: Record<string, unknown> }> | undefined)?.find((n) => n.node_id === job.node_id);
                                                        // Tuning config is a free-form server payload that varies by strategy.
                                                        type TuningConfig = {
                                                            strategy?: string;
                                                            search_strategy?: string;
                                                            strategy_params?: Record<string, unknown>;
                                                            metric?: string;
                                                            n_trials?: number;
                                                            cv_enabled?: boolean;
                                                            cv_type?: string;
                                                            cv_folds?: number;
                                                            cv_shuffle?: boolean;
                                                        };
                                                        const jobConfig = job.config as { tuning_config?: TuningConfig } | undefined;
                                                        const rawConfig = Object.keys(jobConfig?.tuning_config || {}).length > 0
                                                            ? jobConfig?.tuning_config
                                                            : (node?.params?.tuning_config as TuningConfig | undefined);
                                                        const config: TuningConfig | undefined = rawConfig;
                                                        if (!config) return <div className="text-gray-400 col-span-2">No configuration found</div>;
                                                        
                                                        const activeStrategy = config.strategy || config.search_strategy || '';
                                                        const sp = config.strategy_params;
                                                        const hasStrategyParams = sp && Object.keys(sp).length > 0;
                                                        const strategyParamsDisplay = hasStrategyParams
                                                            ? Object.entries(sp!).map(([k, v]) => `${k}: ${v}`).join(' · ')
                                                            : activeStrategy === 'optuna'
                                                            ? 'sampler: tpe · pruner: median (defaults)'
                                                            : activeStrategy === 'halving_grid' || activeStrategy === 'halving_random'
                                                            ? 'factor: 3 · resource: n_samples · min_resources: exhaust (defaults)'
                                                            : '-';

                                                        return (
                                                            <>
                                                                <div>
                                                                    <span className="text-gray-500">Strategy:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300 capitalize">{activeStrategy || '-'}</span>
                                                                    
                                                                </div>
                                                                <div className="col-span-2">
                                                                    <span className="text-gray-500">Strategy Params:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{strategyParamsDisplay}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">Metric:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.metric || '-'}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">Trials:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.n_trials || '-'}</span>
                                                                </div>
                                                                <div>
                                                                    <span className="text-gray-500">CV Enabled:</span>
                                                                    <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_enabled ? 'Yes' : 'No'}</span>
                                                                </div>
                                                                {config.cv_enabled && (
                                                                    <>
                                                                        <div>
                                                                            <span className="text-gray-500">CV Method:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_type || 'Unknown'}</span>
                                                                        </div>
                                                                        <div>
                                                                            <span className="text-gray-500">Folds:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_folds}</span>
                                                                        </div>
                                                                        <div>
                                                                            <span className="text-gray-500">Shuffle:</span>
                                                                            <span className="ml-2 font-mono text-gray-700 dark:text-gray-300">{config.cv_shuffle ? 'Yes' : 'No'}</span>
                                                                        </div>
                                                                    </>
                                                                )}
                                                            </>
                                                        );
                                                    })()}
                                                </div>
                                            </div>
                                        )}

                                        {/* Best Score */}
                                        {(job.result as Record<string, unknown>).best_score !== undefined && (
                                            <div className="p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg w-fit">
                                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                                                    Best Score{getScoringMetric(job) ? ` (${formatMetricName(getScoringMetric(job))})` : ''}
                                                </div>
                                                <div className="font-mono font-bold text-lg text-purple-600 dark:text-purple-400">
                                                    {Number((job.result as Record<string, unknown>).best_score).toFixed(4)}
                                                </div>
                                            </div>
                                        )}

                                        {/* Full Metrics (Train/Test/Val) */}
                                        {!!(job.result as Record<string, unknown>).metrics && (
                                            <div className="space-y-2">
                                                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Evaluation Metrics</h4>
                                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                                    {Object.entries((job.result as Record<string, unknown>).metrics as Record<string, unknown>)
                                                        .filter(([k, v]) => !['best_score', 'best_params', 'trials'].includes(k) && (typeof v === 'number' || typeof v === 'string'))
                                                        .map(([k, v]) => (
                                                        <div key={k} className={`p-3 border rounded-lg ${k.startsWith('cv_') ? 'bg-purple-50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700'}`}>
                                                            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 capitalize">{k.replace(/_/g, ' ')}</div>
                                                            <div className={`font-mono font-medium ${k.startsWith('cv_') ? 'text-purple-600 dark:text-purple-400' : 'text-blue-600 dark:text-blue-400'}`}>
                                                                {typeof v === 'number' ? formatMetricValue(k, v as number) : String(v)}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                            <FeatureImportancesSection result={job.result as Record<string, unknown>} />
                                        
                                        {/* Best Params */}
                                        {!!(job.result as Record<string, unknown>).best_params && (
                                            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-xs overflow-x-auto">
                                                <div className="text-gray-500 mb-2"># Best Hyperparameters</div>
                                                <pre>{JSON.stringify((job.result as Record<string, unknown>).best_params, null, 2)}</pre>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="bg-gray-900 rounded-lg p-4 min-h-[400px] font-mono text-xs text-gray-300 overflow-x-auto">
                        {job.logs && job.logs.length > 0 ? (
                            job.logs.map((log, i) => (
                                <div key={i} className="mb-1 border-b border-gray-800 pb-1 last:border-0">
                                    {log}
                                </div>
                            ))
                        ) : (
                            <div className="text-gray-500 italic">No logs available yet...</div>
                        )}
                        <div ref={logsEndRef} />
                    </div>
                )}
            </div>
        </div>
    );
};

const JobRow: React.FC<{ job: JobInfo; onClick: () => void }> = ({ job, onClick }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50';
      case 'failed': return 'bg-red-50/30 dark:bg-red-900/10 border-red-100 dark:border-red-900/30 hover:bg-red-50/50 dark:hover:bg-red-900/20';
      case 'running': return 'bg-blue-50/30 dark:bg-blue-900/10 border-blue-100 dark:border-blue-900/30 hover:bg-blue-50/50 dark:hover:bg-blue-900/20';
      default: return 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700';
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  };

  const getDuration = (start: string | null, end: string | null) => {
      if (!start || !end) return '-';
      const diff = new Date(end).getTime() - new Date(start).getTime();
      const seconds = Math.floor(diff / 1000);
      if (seconds < 60) return `${seconds}s`;
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ${seconds % 60}s`;
  };

  return (
    <div 
        {...clickableProps(onClick)}
        className={`grid grid-cols-12 gap-4 p-3 rounded-lg border text-sm items-center transition-colors cursor-pointer ${getStatusColor(job.status)}`}
    >
      {/* Status */}
      <div className="col-span-2 flex items-center gap-2">
        <StatusBadge status={job.status} />
      </div>

      {/* Dataset & Model */}
      <div className="col-span-2 flex flex-col justify-center text-xs text-gray-600 dark:text-gray-400 truncate">
        <div className="flex items-center gap-1" title={job.dataset_name || job.dataset_id}>
            <Database className="w-3 h-3" />
            <span className="truncate">{job.dataset_name || job.dataset_id || '-'}</span>
        </div>
        <div className="flex items-center gap-1 mt-0.5 text-[10px] text-gray-500">
            <span className="font-medium truncate">{job.model_type || 'Unknown Model'}</span>
            {job.job_type === 'advanced_tuning' && job.search_strategy && (
                <span className="text-gray-400 truncate">({job.search_strategy})</span>
            )}
        </div>
      </div>

      {/* Job ID */}
      <div className="col-span-3 font-mono text-xs text-gray-500 dark:text-gray-400 break-all" title={job.job_id}>
        {job.job_id}
      </div>

      {/* Started */}
      <div className="col-span-2 text-gray-600 dark:text-gray-400 text-xs">
        {formatDate(job.start_time)}
      </div>

      {/* Duration */}
      <div className="col-span-1 text-gray-600 dark:text-gray-400 text-xs font-mono">
        {getDuration(job.start_time, job.end_time)}
      </div>

      {/* Score */}
      <div className="col-span-2 flex items-center gap-2">
        {job.error ? (
            <span className="text-red-600 dark:text-red-400 text-xs truncate" title={job.error}>
                Error
            </span>
        ) : job.status === 'completed' && job.result ? (
            job.job_type === 'basic_training' && !!(job.result as { metrics?: Record<string, unknown> }).metrics ? (
               <div className="flex flex-wrap gap-1">
                 {Object.entries((job.result as { metrics: Record<string, unknown> }).metrics).slice(0, 1).map(([k, v]) => (
                   <span key={k} className="text-[10px] bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-gray-600 dark:text-gray-300 border border-gray-200 dark:border-gray-600">
                     {k}: {Number(v).toFixed(3)}
                   </span>
                 ))}
               </div>
            ) : job.job_type === 'advanced_tuning' ? (
               <div className="flex flex-wrap gap-1">
                 {(job.result as { best_score?: number }).best_score !== undefined && (
                   <span className="text-[10px] bg-purple-50 dark:bg-purple-900/20 px-1.5 py-0.5 rounded text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                     {formatMetricName(getScoringMetric(job)) || 'Score'}: {Number((job.result as { best_score?: number }).best_score).toFixed(4)}
                   </span>
                 )}
                 {!(job.result as Record<string, unknown>).best_score && !!(job.result as Record<string, unknown>).best_params && (
                   <span className="text-[10px] text-gray-500 dark:text-gray-400">Params found</span>
                 )}
               </div>
            ) : <span className="text-gray-400 text-xs">-</span>
        ) : (
            <span className="text-gray-400 text-xs">-</span>
        )}
      </div>
    </div>
  );
};
