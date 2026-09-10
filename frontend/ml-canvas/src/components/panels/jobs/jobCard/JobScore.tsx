import type { JobInfo } from '../../../../core/api/jobs';
import type { RegistryItem } from '../../../../core/api/registry';
import { formatMetricName, getEnsembleSubTask } from '../../../../core/utils/format';
import { getDisplayScore, getTaskForModelType, hasTuningMetadata, type DisplayScore, type ExperimentsTask } from '../../../pages/ExperimentsPage/utils/jobMeta';

const SPLIT_LABEL: Record<'test' | 'val' | 'train' | 'cv', string> = {
  test: 'test', val: 'val', train: 'train', cv: 'cv',
};

/** The badge distinguishes CV precision from the held-out/train split labels. */
function ScoreBadge({ score }: { score: DisplayScore }) {
  return (
    <div className="flex flex-wrap gap-1">
      <span
        className={`text-[10px] px-1.5 py-0.5 rounded border ${score.split === 'cv'
            ? 'bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-600'
          }`}
        title={`${SPLIT_LABEL[score.split]} split`}
      >
        {formatMetricName(score.metric) || score.metric}: {score.value.toFixed(score.split === 'cv' ? 4 : 3)}
        {score.split !== 'cv' && <span className="opacity-60"> ({SPLIT_LABEL[score.split]})</span>}
      </span>
    </div>
  );
}

/** Only completed error-free jobs are eligible for a headline score. */
function getCardScore(job: JobInfo, registryItems: RegistryItem[]) {
  const task: ExperimentsTask = getTaskForModelType(job.model_type, registryItems);
  // Ensemble jobs are scored on their underlying classification/regression
  // metrics — resolve the effective task for metric-priority lookup so
  // getDisplayScore picks the right list (there is no 'ensemble' entry in
  // SCORE_METRIC_PRIORITY).
  const metricTask: ExperimentsTask = task === 'ensemble' ? (getEnsembleSubTask(job.model_type) ?? 'classification') : task;
  const score = job.status === 'completed' && !job.error ? getDisplayScore(job, metricTask) : null;
  return score;
}

/** Preserve error precedence and the tuned-parameter fallback independently of row identity. */
export function JobScore({ job, registryItems }: { job: JobInfo; registryItems: RegistryItem[] }) {
  const score = getCardScore(job, registryItems);
  if (job.error) {
    return <span className="text-red-600 dark:text-red-400 text-xs truncate" title={job.error}>Error</span>;
  }
  if (job.status === 'completed' && job.result) {
    if (score) return <ScoreBadge score={score} />;
    if (hasTuningMetadata(job) && !!(job.result as Record<string, unknown>).best_params) {
      return <span className="text-[10px] text-gray-500 dark:text-gray-400">Params found</span>;
    }
  }
  return <span className="text-gray-400 text-xs">-</span>;
}
