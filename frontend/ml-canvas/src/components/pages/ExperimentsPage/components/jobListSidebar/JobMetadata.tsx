import { GitBranch, Trophy } from 'lucide-react';
import type { JobInfo } from '../../../../../core/api/jobs';
import { hasTuningMetadata } from '../../utils/jobMeta';

/** Model, dataset and run badges shown on an expanded row. */
export function JobMetadata({ job }: { job: JobInfo }) {
  return (
    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
      {job.model_type} • {job.dataset_name || 'Unknown Dataset'}
      {hasTuningMetadata(job) && (job.search_strategy || (job.config as { tuning?: { strategy?: string } }).tuning?.strategy) && (
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
    </div>);
}
