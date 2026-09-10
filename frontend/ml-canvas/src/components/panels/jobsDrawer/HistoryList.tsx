import { ChevronDown, RefreshCw } from 'lucide-react';
import type { JobInfo } from '../../../core/api/jobs';
import type { RegistryItem } from '../../../core/api/registry';
import type { SubmittedRun } from '../../../core/types/runFeedback';
import { VirtualList } from '../../shared/VirtualList';
import { JobCard } from '../jobs/JobCard';

interface HistoryListProps {
  filteredJobs: JobInfo[];
  registryItems: RegistryItem[];
  setSelectedJob: (job: JobInfo) => void;
  hasMore: boolean;
  inspectedRun: SubmittedRun | null;
  isLoading: boolean;
  loadMoreJobs: () => Promise<void>;
  emptyMessage: string;
}

/** Keep history order, virtual row keys and pagination presentation together. */
export function HistoryList({ filteredJobs, registryItems, setSelectedJob, hasMore,
  inspectedRun, isLoading, loadMoreJobs, emptyMessage }: HistoryListProps) {
  return <>
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
        <div className="text-center py-10 text-gray-500 dark:text-gray-400 text-sm">
          {emptyMessage}
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
                <JobCard job={job} onClick={() => { setSelectedJob(job); }} registryItems={registryItems} />
              </div>
            )}
          />

          {hasMore && !inspectedRun && (
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
  </>;
}
