import React from 'react';
import { Database } from 'lucide-react';
import { JobInfo } from '../../../core/api/jobs';
import { clickableProps } from '../../../core/utils/a11y';
import { formatMetricName, formatDuration, isEnsembleModelType, getEnsembleSubTask, getEnsembleStrategy } from '../../../core/utils/format';
import { StatusBadge } from '../../shared/StatusBadge';
import { getDisplayScore, getTaskForModelType, type ExperimentsTask } from '../../pages/ExperimentsPage/utils/jobMeta';
import type { RegistryItem } from '../../../core/api/registry';

const SPLIT_LABEL: Record<'test' | 'val' | 'train' | 'cv', string> = {
  test: 'test', val: 'val', train: 'train', cv: 'cv',
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

interface JobCardProps {
  job: JobInfo;
  onClick: () => void;
  registryItems: RegistryItem[];
}

export const JobCard: React.FC<JobCardProps> = ({ job, onClick, registryItems }) => {
  const task: ExperimentsTask = getTaskForModelType(job.model_type, registryItems);
  // Ensemble jobs are scored on their underlying classification/regression
  // metrics — resolve the effective task for metric-priority lookup so
  // getDisplayScore picks the right list (there is no 'ensemble' entry in
  // SCORE_METRIC_PRIORITY).
  const metricTask: ExperimentsTask = task === 'ensemble' ? (getEnsembleSubTask(job.model_type) ?? 'classification') : task;
  const score = job.status === 'completed' && !job.error ? getDisplayScore(job, metricTask) : null;
  const isEnsemble = isEnsembleModelType(job.model_type);
  const ensembleStrategy = getEnsembleStrategy(job.model_type);
  const ensembleSubTask = getEnsembleSubTask(job.model_type);

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
      <div className="flex items-center gap-1 mt-0.5 text-[10px] text-gray-500 flex-wrap">
        <span className="font-medium truncate">{job.model_type || 'Unknown Model'}</span>
        {job.job_type === 'tuning' && job.search_strategy && (
          <span className="text-gray-400 truncate">({job.search_strategy})</span>
        )}
        {isEnsemble && (
          <span className="px-1.5 py-0.5 rounded border bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800 whitespace-nowrap">
            {ensembleStrategy} · {ensembleSubTask === 'regression' ? 'Regression' : 'Classification'}
          </span>
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
      {formatDuration(job.start_time, job.end_time)}
    </div>

    {/* Score */}
    <div className="col-span-2 flex items-center gap-2">
      {job.error ? (
          <span className="text-red-600 dark:text-red-400 text-xs truncate" title={job.error}>
              Error
          </span>
      ) : job.status === 'completed' && job.result ? (
          score ? (
             <div className="flex flex-wrap gap-1">
               <span
                 className={`text-[10px] px-1.5 py-0.5 rounded border ${
                   score.split === 'cv'
                     ? 'bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800'
                     : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-600'
                 }`}
                 title={`${SPLIT_LABEL[score.split]} split`}
               >
                 {formatMetricName(score.metric) || score.metric}: {score.value.toFixed(score.split === 'cv' ? 4 : 3)}
                 {score.split !== 'cv' && <span className="opacity-60"> ({SPLIT_LABEL[score.split]})</span>}
               </span>
             </div>
          ) : job.job_type === 'tuning' && !!(job.result as Record<string, unknown>).best_params ? (
             <span className="text-[10px] text-gray-500 dark:text-gray-400">Params found</span>
          ) : <span className="text-gray-400 text-xs">-</span>
      ) : (
          <span className="text-gray-400 text-xs">-</span>
      )}
    </div>
  </div>
  );
};
