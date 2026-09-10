import { Database } from 'lucide-react';
import type { JobInfo } from '../../../../core/api/jobs';
import { isEnsembleModelType, getEnsembleSubTask, getEnsembleStrategy } from '../../../../core/utils/format';
import { hasTuningMetadata } from '../../../pages/ExperimentsPage/utils/jobMeta';

/** Dataset, model and training badges retain their existing fallback labels. */
export function JobIdentity({ job }: { job: JobInfo }) {
  const isEnsemble = isEnsembleModelType(job.model_type);
  const ensembleStrategy = getEnsembleStrategy(job.model_type);
  const ensembleSubTask = getEnsembleSubTask(job.model_type);
  return (
    <div className="col-span-2 flex flex-col justify-center text-xs text-gray-600 dark:text-gray-400 truncate">
      <div className="flex items-center gap-1" title={job.dataset_name || job.dataset_id}>
        <Database className="w-3 h-3" />
        <span className="truncate">{job.dataset_name || job.dataset_id || '-'}</span>
      </div>
      <div className="flex items-center gap-1 mt-0.5 text-[10px] text-gray-500 flex-wrap">
        <span className="font-medium truncate">{job.model_type || 'Unknown Model'}</span>
        {hasTuningMetadata(job) && job.search_strategy && (
          <span className="text-gray-400 truncate">({job.search_strategy})</span>
        )}
        {job.engine === 'polars' && (
          <span
            className="px-1.5 py-0.5 rounded border bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-800 whitespace-nowrap"
            title="Trained on the Polars engine"
          >
            Polars
          </span>
        )}
        {isEnsemble && (
          <span className="px-1.5 py-0.5 rounded border bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800 whitespace-nowrap">
            {ensembleStrategy} · {ensembleSubTask === 'regression' ? 'Regression' : 'Classification'}
          </span>
        )}
      </div>
    </div>
  );
}
