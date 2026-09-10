import React from 'react';
import { JobInfo } from '../../../core/api/jobs';
import { clickableProps } from '../../../core/utils/a11y';
import { formatDuration } from '../../../core/utils/format';
import { StatusBadge } from '../../shared/StatusBadge';
import type { RegistryItem } from '../../../core/api/registry';

import { JobIdentity } from './jobCard/JobIdentity';
import { JobScore } from './jobCard/JobScore';

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50';
    case 'failed':
      return 'bg-red-50/30 dark:bg-red-900/10 border-red-100 dark:border-red-900/30 hover:bg-red-50/50 dark:hover:bg-red-900/20';
    case 'running':
      return 'bg-blue-50/30 dark:bg-blue-900/10 border-blue-100 dark:border-blue-900/30 hover:bg-blue-50/50 dark:hover:bg-blue-900/20';
    default:
      return 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700';
  }
};

const formatDate = (dateStr: string | null): string => {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

interface JobCardProps {
  job: JobInfo;
  onClick: () => void;
  registryItems: RegistryItem[];
}

export const JobCard: React.FC<JobCardProps> = ({ job, onClick, registryItems }) => {

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
      <JobIdentity job={job} />

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
        {formatDuration(job.start_time, job.end_time)}
      </div>

      {/* Score */}
      <div className="col-span-2 flex items-center gap-2">
        <JobScore job={job} registryItems={registryItems} />
      </div>
    </div>
  );
};
