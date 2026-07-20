import React, { useState, useEffect, useRef } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { X, RefreshCw, ChevronDown, Zap, CheckCircle2, Search, Filter } from 'lucide-react';
import { JobInfo } from '../../core/api/jobs';
import { RegistryItem, registryApi } from '../../core/api/registry';
import { getTaskForModelType } from '../pages/ExperimentsPage/utils/jobMeta';
import { TaskType } from '../../core/types/taskType';
import { useEscapeKey } from '../../core/hooks/useEscapeKey';
import { VirtualList } from '../shared/VirtualList';
import { JobCard } from './jobs/JobCard';
import { JobDetailsView } from './jobs/JobDetailsView';

/** Job History tabs, in display order (plan §0.5: task type, not engine). */
const TASK_TABS: { task: TaskType; label: string }[] = [
  { task: 'classification', label: 'Classification' },
  { task: 'regression', label: 'Regression' },
  { task: 'text_classification', label: 'Text Classification' },
  { task: 'segmentation', label: 'Segmentation' },
];

const TASK_LABELS: Record<TaskType, string> = {
  classification: 'classification',
  regression: 'regression',
  text_classification: 'text classification',
  segmentation: 'segmentation',
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
  const [registryItems, setRegistryItems] = useState<RegistryItem[]>([]);

  // One-time fetch of registry items (mirrors ExperimentsPage.tsx's
  // fetchDatasets()-style pattern) so job task types can be resolved via
  // getTaskForModelType. Not cached/shared across components — kept as a
  // simple local fetch to minimize risk of this change.
  useEffect(() => {
    let cancelled = false;
    registryApi.getAllNodes()
      .then(nodes => { if (!cancelled) setRegistryItems(nodes); })
      .catch(error => { console.error('Failed to fetch registry items:', error); });
    return () => { cancelled = true; };
  }, []);

  const panelRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const titleId = 'jobs-drawer-title';

  useEscapeKey(toggleDrawer, isDrawerOpen);

  // Reset to list view whenever the drawer re-opens
  useEffect(() => {
    if (isDrawerOpen) setSelectedJob(null);
  }, [isDrawerOpen]);

  // Focus management: move focus into the panel when the drawer opens so
  // keyboard/screen-reader users land inside it, and restore focus to
  // whatever triggered it on close. Kept minimal (no full focus trap).
  useEffect(() => {
    if (!isDrawerOpen) return;
    previouslyFocusedRef.current = (document.activeElement as HTMLElement | null) ?? null;
    const raf = window.requestAnimationFrame(() => {
      panelRef.current?.focus();
    });
    return () => {
      window.cancelAnimationFrame(raf);
      const prev = previouslyFocusedRef.current;
      if (prev && document.contains(prev)) {
        try {
          prev.focus();
        } catch {
          // Ignore: element may have become un-focusable mid-flight.
        }
      }
    };
  }, [isDrawerOpen]);

  // Auto-load more when the current tab shows fewer than 5 jobs but the
  // server still has more.  The store fetches all jobs together regardless
  // of task, so a tab that is sparse (e.g. 2 segmentation jobs among 50
  // classification jobs) keeps fetching until it reaches the threshold or
  // exhausts the server.
  //
  // Cap the number of consecutive auto-triggered page fetches: if the
  // active tab's task is very rare relative to total volume, this
  // effect would otherwise re-fire on every `jobs` update and hammer the
  // API fetching dozens of pages back-to-back trying to reach the
  // threshold. Once the cap is hit we stop auto-loading for this tab —
  // the user can still click "Load More History" manually. The counter
  // resets whenever the user switches tabs or reopens the drawer, so a
  // legitimate tab switch isn't permanently blocked by an earlier cap-out.
  const MAX_AUTO_LOAD_ATTEMPTS = 5;
  const autoLoadAttemptsRef = useRef(0);

  useEffect(() => {
    autoLoadAttemptsRef.current = 0;
  }, [isDrawerOpen, activeTab]);

  useEffect(() => {
    if (!isDrawerOpen || isLoading || !hasMore) return;
    if (autoLoadAttemptsRef.current >= MAX_AUTO_LOAD_ATTEMPTS) return;
    const tabCount = jobs.filter(j => getTaskForModelType(j.model_type, registryItems) === activeTab).length;
    if (tabCount < 5) {
      autoLoadAttemptsRef.current += 1;
      void loadMoreJobs();
    }
  }, [isDrawerOpen, activeTab, jobs, hasMore, isLoading, loadMoreJobs, registryItems]);

  if (!isDrawerOpen) return null;

  const tabJobs = jobs.filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab);

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
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className="relative w-[1200px] max-w-[95vw] h-[85vh] bg-white dark:bg-gray-800 shadow-2xl rounded-lg flex flex-col border border-gray-200 dark:border-gray-700 overflow-hidden transition-all outline-none"
      >

        {selectedJob ? (
            <JobDetailsView job={selectedJob} onBack={() => { setSelectedJob(null); }} onClose={() => toggleDrawer(false)} />
        ) : (
            <>
                {/* Header */}
                <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
                <h2 id={titleId} className="font-semibold text-gray-800 dark:text-gray-100">Job History</h2>
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
                {TASK_TABS.map(({ task, label }) => (
                    <button
                        key={task}
                        className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${
                        activeTab === task
                            ? 'border-blue-500 text-blue-600 dark:text-blue-400 bg-blue-50/50 dark:bg-blue-900/20'
                            : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                        }`}
                        onClick={() => setTab(task)}
                    >
                        {label}
                    </button>
                ))}
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
                      : `No ${TASK_LABELS[activeTab]} jobs found.`
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
                                <JobCard job={job} onClick={() => { setSelectedJob(job); }} />
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
