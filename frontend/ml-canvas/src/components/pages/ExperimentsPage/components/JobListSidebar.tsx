import React from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, RefreshCw, Rocket, Trophy, GitBranch } from 'lucide-react';
import type { JobInfo } from '../../../../core/api/jobs';
import { clickableProps } from '../../../../core/utils/a11y';
import { shortRunId } from '../utils/jobMeta';

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
                          title={job.promoted_at ? 'Unpromote' : 'Promote as Winner'}
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
              {!isSidebarCollapsed && 'Load More Runs'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
