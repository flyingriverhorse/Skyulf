import React, { useState, useEffect, useRef } from 'react';
import { useJobStore } from '../../core/store/useJobStore';
import { X, RefreshCw } from 'lucide-react';
import { JobInfo } from '../../core/api/jobs';
import { RegistryItem, registryApi } from '../../core/api/registry';
import { getTaskForModelType } from '../pages/ExperimentsPage/utils/jobMeta';
import { useEscapeKey } from '../../core/hooks/useEscapeKey';
import { JobDetailsView } from './jobs/JobDetailsView';
import { HistoryTabs } from './jobsDrawer/HistoryTabs';
import { HistoryFilters } from './jobsDrawer/HistoryFilters';
import { HistoryList } from './jobsDrawer/HistoryList';
import { ParallelRunProgress } from './jobsDrawer/ParallelRunProgress';
import { selectHistoryJobs, filterHistoryJobs, getHistoryFacets, getRunProgress, getEmptyHistoryMessage } from './jobsDrawer/history';

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
    activeParallelRun,
    inspectedRun,
    setInspectedRun,
    runJobs,
  } = useJobStore();

  const [selectedJob, setSelectedJob] = useState<JobInfo | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [ensembleSubFilter, setEnsembleSubFilter] = useState<'all' | 'classification' | 'regression'>('all');
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
  }, [isDrawerOpen, inspectedRun]);

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
    if (activeTab !== 'ensemble') setEnsembleSubFilter('all');
  }, [activeTab]);

  useEffect(() => {
    if (!isDrawerOpen || isLoading || !hasMore || inspectedRun) return;
    if (autoLoadAttemptsRef.current >= MAX_AUTO_LOAD_ATTEMPTS) return;
    const tabCount = jobs.filter(j => getTaskForModelType(j.model_type, registryItems) === activeTab).length;
    if (tabCount < 5) {
      autoLoadAttemptsRef.current += 1;
      void loadMoreJobs();
    }
  }, [isDrawerOpen, activeTab, jobs, hasMore, isLoading, loadMoreJobs, registryItems, inspectedRun]);

  if (!isDrawerOpen) return null;

  const filters = { searchQuery, statusFilter, modelFilter };
  const tabJobs = selectHistoryJobs({ jobs, runJobs, inspectedRun, activeTab, ensembleSubFilter, registryItems });
  const { modelTypes, statuses } = getHistoryFacets(tabJobs);
  const filteredJobs = filterHistoryJobs(tabJobs, filters, inspectedRun);
  const progress = getRunProgress(activeParallelRun, inspectedRun, runJobs, jobs);
  const emptyMessage = getEmptyHistoryMessage(inspectedRun, filters, activeTab);

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
                  aria-label="Close job history"
                  className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Parallel Run Progress Banner */}
            <ParallelRunProgress progress={progress} />

            {/* Tabs */}
            {inspectedRun ? <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-b p-4 text-sm">
              <span className="min-w-0 break-words">{inspectedRun.label} · {inspectedRun.jobIds.length} submitted jobs</span>
              <button type="button" onClick={() => setInspectedRun(null)} className="rounded text-primary underline underline-offset-2 focus-ring">Show all jobs</button>
            </div> : <>
              <HistoryTabs activeTab={activeTab} setTab={setTab}
                ensembleSubFilter={ensembleSubFilter} setEnsembleSubFilter={setEnsembleSubFilter} />
              <HistoryFilters {...filters} setSearchQuery={setSearchQuery}
                setStatusFilter={setStatusFilter} setModelFilter={setModelFilter}
                showFilters={showFilters} setShowFilters={setShowFilters}
                statuses={statuses} modelTypes={modelTypes} />
            </>}

            <HistoryList filteredJobs={filteredJobs} registryItems={registryItems}
              setSelectedJob={setSelectedJob} hasMore={hasMore} inspectedRun={inspectedRun}
              isLoading={isLoading} loadMoreJobs={loadMoreJobs} emptyMessage={emptyMessage} />
          </>
        )}
      </div>
    </div>
  );
};
