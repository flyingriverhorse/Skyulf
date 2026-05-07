import React from 'react';
import { Database } from 'lucide-react';
import { JobInfo } from '../../../core/api/jobs';
import { clickableProps } from '../../../core/utils/a11y';
import { formatMetricName } from '../../../core/utils/format';
import { StatusBadge } from '../../shared/StatusBadge';

/** Extract the scoring metric name from a job's result or config. */
const getScoringMetric = (job: JobInfo): string | undefined => {
  const result = job.result as Record<string, unknown> | undefined;
  if (result?.scoring_metric) return result.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  return tuning?.metric as string | undefined;
};

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

const getDuration = (start: string | null, end: string | null): string => {
  if (!start || !end) return '-';
  const diff = new Date(end).getTime() - new Date(start).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
};

interface JobCardProps {
  job: JobInfo;
  onClick: () => void;
}

export const JobCard: React.FC<JobCardProps> = ({ job, onClick }) => (
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
