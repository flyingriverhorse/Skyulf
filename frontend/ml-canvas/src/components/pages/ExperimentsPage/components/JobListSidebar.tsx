import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { JobRow } from './jobListSidebar/JobRow';
import { JobPagination } from './jobListSidebar/JobPagination';
import type { JobInfo } from '../../../../core/api/jobs';

interface Props {
  filteredJobs: JobInfo[];
  selectedJobIds: string[];
  isSidebarCollapsed: boolean;
  setIsSidebarCollapsed: (v: boolean) => void;
  toggleJobSelection: (jobId: string) => void;
  hasMore: boolean;
  isLoading: boolean;
  loadMoreJobs: () => void | Promise<void>;
  handlePromote: (e: React.MouseEvent, job: JobInfo) => void | Promise<void>;
  handleDeploy: (e: React.MouseEvent, jobId: string) => void | Promise<void>;
  getDuration: (start: string | null, end: string | null) => string;
}

export const JobListSidebar: React.FC<Props> = ({
  filteredJobs,
  selectedJobIds,
  isSidebarCollapsed,
  setIsSidebarCollapsed,
  toggleJobSelection,
  hasMore,
  isLoading,
  loadMoreJobs,
  handlePromote,
  handleDeploy,
  getDuration,
}) => {
  return (
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
          title={isSidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {isSidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {filteredJobs.length === 0 && !isLoading && !isSidebarCollapsed && (
          <div className="p-4 text-center text-xs text-gray-500 dark:text-gray-400">
            No runs match the current filters.
          </div>
        )}
        {filteredJobs.map(job => (
          <JobRow key={job.job_id} job={job} selectedJobIds={selectedJobIds} isSidebarCollapsed={isSidebarCollapsed} toggleJobSelection={toggleJobSelection} handlePromote={handlePromote} handleDeploy={handleDeploy} getDuration={getDuration} />
        ))}

        <JobPagination hasMore={hasMore} isSidebarCollapsed={isSidebarCollapsed} isLoading={isLoading} loadMoreJobs={loadMoreJobs} />
      </div>
    </div>
  );
};
