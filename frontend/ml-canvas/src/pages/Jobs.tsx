import React, { useEffect, useState } from 'react';
import {
  Activity, CheckCircle, XCircle, Clock, Search,
  RefreshCw, Database, BarChart2, Filter, Tags, TrendingUp, FileText, Boxes, Layers
} from 'lucide-react';
import { jobsApi, JobInfo } from '../core/api/jobs';
import { registryApi, RegistryItem } from '../core/api/registry';
import { getTaskForModelType } from '../components/pages/ExperimentsPage/utils/jobMeta';
import { getEnsembleSubTask } from '../core/utils/format';
import type { TaskType } from '../core/types/taskType';
import { LoadingState, EmptyState } from '../components/shared';
import { formatDuration } from '../core/utils/format';

type NonTaskTab = 'eda' | 'ingestion';
type TabType = TaskType | NonTaskTab;

/** Job History-style task tabs, in display order (mirrors JobsDrawer.tsx/ExperimentsPage.tsx). */
const TASK_TABS: { task: TaskType; label: string; icon: React.ReactNode }[] = [
  { task: 'classification', label: 'Classification', icon: <Tags size={16} /> },
  { task: 'regression', label: 'Regression', icon: <TrendingUp size={16} /> },
  { task: 'text_classification', label: 'Text Classification', icon: <FileText size={16} /> },
  { task: 'segmentation', label: 'Segmentation', icon: <Boxes size={16} /> },
  { task: 'ensemble', label: 'Ensemble', icon: <Layers size={16} /> },
];
const TASK_TYPES: TaskType[] = TASK_TABS.map(t => t.task);

const isTaskTab = (tab: TabType): tab is TaskType => (TASK_TYPES as string[]).includes(tab);

export const JobsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('classification');
  const [registryItems, setRegistryItems] = useState<RegistryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [ensembleSubFilter, setEnsembleSubFilter] = useState<'all' | 'classification' | 'regression'>('all');
  const LIMIT = 25;  // match drawer PAGE_SIZE so both show the same number of rows per tab

  // Backend only filters jobs by job_type (basic_training/advanced_tuning),
  // not by task. So the four task tabs above all share one merged
  // training+tuning "pool" (job_type omitted \u2192 backend merge-sorts both
  // tables, see JobManager.list_jobs), which is filtered client-side by
  // task via getTaskForModelType \u2014 the same pattern useJobStore/JobsDrawer
  // use for the Job History drawer.
  const [pool, setPool] = useState<JobInfo[]>([]);
  const [poolSkip, setPoolSkip] = useState(0);
  const [poolHasMore, setPoolHasMore] = useState(true);

  // EDA and Ingestion tabs are unrelated to task filtering and keep their
  // original per-tab cached pagination untouched.
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [cache, setCache] = useState<Record<NonTaskTab, { data: JobInfo[], skip: number, hasMore: boolean }>>({
    eda: { data: [], skip: 0, hasMore: true },
    ingestion: { data: [], skip: 0, hasMore: true }
  });

  // One-time fetch of the node registry so job task types can be resolved
  // from each job's model_type (mirrors ExperimentsPage.tsx/JobsDrawer.tsx).
  useEffect(() => {
    let cancelled = false;
    registryApi.getAllNodes()
      .then(nodes => { if (!cancelled) setRegistryItems(nodes); })
      .catch(error => { console.error('Failed to fetch node registry:', error); });
    return () => { cancelled = true; };
  }, []);

  const fetchPool = async (reset: boolean = false) => {
    if (reset) {
      setLoading(true);
      setPool([]);
      setPoolSkip(0);
      setPoolHasMore(true);
    }

    try {
      const currentSkip = reset ? 0 : poolSkip;
      // job_type omitted \u2192 backend merges & sorts training + tuning jobs together.
      const fetchedJobs = await jobsApi.getJobs(LIMIT, currentSkip);
      fetchedJobs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      const isMore = fetchedJobs.length >= LIMIT;
      setPoolHasMore(isMore);
      setPool(prev => reset ? fetchedJobs : [...prev, ...fetchedJobs]);
      setPoolSkip(currentSkip + LIMIT);
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchJobs = async (reset: boolean = false) => {
    if (reset) {
      setLoading(true);
      setJobs([]);
      setSkip(0);
      setHasMore(true);
    }

    try {
      const currentSkip = reset ? 0 : skip;
      let fetchedJobs: JobInfo[] = [];

      if (activeTab === 'eda') {
        fetchedJobs = await jobsApi.getEDAJobs(LIMIT);
      } else if (activeTab === 'ingestion') {
        fetchedJobs = await jobsApi.getIngestionJobs(LIMIT, currentSkip);
      }

      // Sort by created_at desc
      fetchedJobs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

      const isMore = fetchedJobs.length >= LIMIT;
      if (!isMore) {
        setHasMore(false);
      }

      if (reset) {
        setJobs(fetchedJobs);
        setCache(prev => ({
          ...prev,
          [activeTab as NonTaskTab]: { data: fetchedJobs, skip: currentSkip + LIMIT, hasMore: isMore }
        }));
      } else {
        setJobs(prev => {
          const updated = [...prev, ...fetchedJobs];
          setCache(prevCache => ({
            ...prevCache,
            [activeTab as NonTaskTab]: { data: updated, skip: currentSkip + LIMIT, hasMore: isMore }
          }));
          return updated;
        });
      }

      setSkip(currentSkip + LIMIT);
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isTaskTab(activeTab)) {
      if (pool.length === 0) {
        void fetchPool(true);
      } else {
        setLoading(false);
      }
      return;
    }

    const cached = cache[activeTab];
    if (cached.data.length > 0) {
      setJobs(cached.data);
      setSkip(cached.skip);
      setHasMore(cached.hasMore);
      setLoading(false);
    } else {
      void fetchJobs(true);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Auto-load more pool pages (capped) if the active task tab has too few
  // matching jobs yet, mirroring JobsDrawer.tsx's identical safeguard
  // against hammering the API when a task is rare relative to total volume.
  const MAX_AUTO_LOAD_ATTEMPTS = 5;
  const [autoLoadAttempts, setAutoLoadAttempts] = useState(0);

  useEffect(() => {
    setAutoLoadAttempts(0);
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== 'ensemble') setEnsembleSubFilter('all');
  }, [activeTab]);

  useEffect(() => {
    if (!isTaskTab(activeTab) || loading || !poolHasMore) return;
    if (autoLoadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) return;
    const tabCount = pool.filter(j => getTaskForModelType(j.model_type, registryItems) === activeTab).length;
    if (tabCount < LIMIT) {
      setAutoLoadAttempts(prev => prev + 1);
      void fetchPool(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, pool, poolHasMore, loading, registryItems]);

  const handleLoadMore = () => {
    if (isTaskTab(activeTab)) {
      void fetchPool(false);
    } else {
      void fetchJobs(false);
    }
  };

  const visibleJobs = isTaskTab(activeTab)
    ? pool
        .filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab)
        .filter(job => activeTab !== 'ensemble' || ensembleSubFilter === 'all' || getEnsembleSubTask(job.model_type) === ensembleSubFilter)
    : jobs;
  const visibleHasMore = isTaskTab(activeTab) ? poolHasMore : hasMore;

  const filteredJobs = visibleJobs.filter(job => {
    if (statusFilter !== 'all' && job.status.toLowerCase() !== statusFilter) return false;
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      return (
        job.job_id.toLowerCase().includes(term) ||
        (job.model_type && job.model_type.toLowerCase().includes(term)) ||
        (job.dataset_name && job.dataset_name.toLowerCase().includes(term))
      );
    }
    return true;
  });

  const activeFilterCount = (statusFilter !== 'all' ? 1 : 0);

  return (
    <div className="p-8 space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Jobs</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">Monitor and manage all your pipeline activities</p>
        </div>
        <button
          onClick={() => (isTaskTab(activeTab) ? fetchPool(true) : fetchJobs(true))}
          className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-700 dark:text-slate-300"
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>

      {/* Tabs & Filters */}
      <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-white dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
        <div className="flex gap-2 overflow-x-auto w-full md:w-auto pb-2 md:pb-0">
          {TASK_TABS.map(({ task, label, icon }) => (
            <TabButton key={task} active={activeTab === task} onClick={() => setActiveTab(task)} icon={icon} label={label} />
          ))}
          <TabButton active={activeTab === 'eda'} onClick={() => setActiveTab('eda')} icon={<BarChart2 size={16} />} label="EDA" />
          <TabButton active={activeTab === 'ingestion'} onClick={() => setActiveTab('ingestion')} icon={<Database size={16} />} label="Ingestion" />
        </div>

        <div className="flex items-center gap-2">
          <div className="relative w-full md:w-56">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none z-10" size={16} />
            <input
              type="text"
              placeholder="Search jobs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-4 py-2 text-sm bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:text-slate-100"
            />
          </div>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-1.5 px-3 py-2 text-sm rounded-md border transition-colors ${
              showFilters || statusFilter !== 'all'
                ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300'
                : 'bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            <Filter size={14} />
            Filters
            {activeFilterCount > 0 && (
              <span className="ml-0.5 w-5 h-5 flex items-center justify-center rounded-full bg-indigo-600 text-white text-xs">{activeFilterCount}</span>
            )}
          </button>
          {(searchTerm || statusFilter !== 'all') && (
            <button
              onClick={() => { setSearchTerm(''); setStatusFilter('all'); }}
              className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline whitespace-nowrap"
            >
              Clear all
            </button>
          )}
        </div>
      </div>

      {activeTab === 'ensemble' && (
        <div className="flex items-center gap-1.5 px-4 py-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
          {(['all', 'classification', 'regression'] as const).map((value) => (
            <button
              key={value}
              onClick={() => setEnsembleSubFilter(value)}
              className={`px-2.5 py-1 text-xs rounded-full border transition-colors capitalize ${
                ensembleSubFilter === value
                  ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300'
                  : 'bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              {value}
            </button>
          ))}
        </div>
      )}

      {showFilters && (
        <div className="flex items-center gap-4 px-4 py-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">Status</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-2.5 py-1.5 text-sm rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value="all">All statuses</option>
            {[...new Set(visibleJobs.map(j => j.status.toLowerCase()))].sort().map(s => (
              <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
            ))}
          </select>
        </div>
      )}

      {/* Jobs Table */}
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
        {loading ? (
          <LoadingState message="Loading jobs..." />
        ) : filteredJobs.length === 0 ? (
          <EmptyState title="No jobs found matching your criteria." />
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
              <thead className="bg-slate-50 dark:bg-slate-900/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Job ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Type</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Details</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Created At</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-slate-800 divide-y divide-slate-200 dark:divide-slate-700">
                {filteredJobs.map((job) => (
                  <tr key={job.job_id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-slate-100">
                      <span className="font-mono text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
                        {job.job_id.substring(0, 8)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400 capitalize">
                      {job.job_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                      {job.model_type || job.dataset_name || job.target_column || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400 font-mono">
                      {formatDuration(job.start_time, job.end_time)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                      {new Date(job.created_at).toLocaleDateString()} <span className="text-xs opacity-70">{new Date(job.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Load More Button */}
        {!loading && visibleHasMore && filteredJobs.length > 0 && (
          <div className="p-4 border-t border-slate-200 dark:border-slate-700 flex justify-center">
            <button
              onClick={handleLoadMore}
              className="text-sm text-indigo-600 hover:text-indigo-700 font-medium flex items-center gap-1 px-4 py-2 rounded-md hover:bg-indigo-50 dark:hover:bg-indigo-900/30 transition-colors"
            >
              Load More
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const TabButton = ({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all whitespace-nowrap ${
      active
        ? 'bg-indigo-600 text-white shadow-sm'
        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
    }`}
  >
    {icon}
    {label}
  </button>
);

const StatusBadge = ({ status }: { status: string }) => {
  const styles = {
    succeeded: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    completed: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    queued: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-400',
  };

  const icons = {
    succeeded: <CheckCircle size={14} />,
    completed: <CheckCircle size={14} />,
    failed: <XCircle size={14} />,
    running: <Activity size={14} className="animate-pulse" />,
    pending: <Clock size={14} />,
    queued: <Clock size={14} />,
  };

  const key = status.toLowerCase() as keyof typeof styles;
  const style = styles[key] || 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-400';
  const icon = icons[key] || <Clock size={14} />;

  return (
    <span className={`px-2.5 py-1 inline-flex items-center gap-1.5 text-xs font-medium rounded-full ${style}`}>
      {icon}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};
