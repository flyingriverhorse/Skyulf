import React from 'react';
import { JobActions } from './JobActions';
import type { JobInfo } from '../../../../../core/api/jobs';
import { clickableProps } from '../../../../../core/utils/a11y';
import { shortRunId } from '../../utils/jobMeta';
import { JobMetadata } from './JobMetadata';
import type { JobListSidebar } from '../JobListSidebar';
type SidebarProps = React.ComponentProps<typeof JobListSidebar>;

/** A selectable run with its promotion and deployment actions. */
export function JobRow({ job, selectedJobIds, isSidebarCollapsed, toggleJobSelection, handlePromote, handleDeploy, getDuration }: Pick<SidebarProps, 'selectedJobIds' | 'isSidebarCollapsed' | 'toggleJobSelection' | 'handlePromote' | 'handleDeploy' | 'getDuration'> & { job: JobInfo }) {
  return (
    <div
      {...clickableProps(() => { toggleJobSelection(job.job_id); })}
      className={`border-b border-gray-100 dark:border-gray-700 cursor-pointer transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 ${selectedJobIds.includes(job.job_id) ? 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-l-blue-500' : 'border-l-4 border-l-transparent'
        } ${isSidebarCollapsed ? 'p-2 flex justify-center' : 'p-3'}`}
      title={isSidebarCollapsed ? `${shortRunId(job)} · ${job.model_type}` : undefined}
    >
      {isSidebarCollapsed ? (
        <div className={`w-2 h-2 rounded-full ${job.status === 'completed' ? 'bg-green-500' :
            job.status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
          }`} />
      ) : (
        <>
          <div className="flex justify-between items-start mb-1">
            <span className="font-mono text-xs font-semibold text-gray-700 dark:text-gray-300 break-all">
              {shortRunId(job)}
            </span>
            <div className="flex items-center gap-2">
              <JobActions job={job} handlePromote={handlePromote} handleDeploy={handleDeploy} />
            </div>
          </div>
          <JobMetadata job={job} />
          <div className="flex justify-between items-center text-[10px] text-gray-400">
            <span>{new Date(job.start_time || job.created_at).toLocaleString()}</span>
            <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-gray-600 dark:text-gray-300">
              {getDuration(job.start_time, job.end_time)}
            </span>
          </div>
        </>
      )}
    </div>);
}
