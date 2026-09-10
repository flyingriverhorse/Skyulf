import type { JobInfo } from '../../../../../core/api/jobs';
import { isEnsembleModelType, extractEnsembleSummary, formatBaseEstimator } from '../../../../../core/utils/format';
import { hasTuningMetadata } from '../../utils/jobMeta';
import { getJobConfig } from './config';

const summaryFor = (job: JobInfo) => {
  if (!isEnsembleModelType(job.model_type)) return null;
  let bucket: Record<string, unknown> | undefined;
  if (hasTuningMetadata(job)) {
    const cfg = getJobConfig(job);
    bucket = cfg?.tuning_config as Record<string, unknown> | undefined;
  } else {
    const hp = job.hyperparameters as Record<string, unknown> | undefined;
    const nested = hp?.hyperparameters;
    bucket = (nested && typeof nested === 'object' && !Array.isArray(nested))
      ? nested as Record<string, unknown>
      : hp;
  }
  return extractEnsembleSummary(job.model_type, bucket);
};

/** Resolve ensemble structure once per selected job set. */
export function prepareEnsembleData(selectedJobs: JobInfo[]) {
  if (!selectedJobs.some(job => isEnsembleModelType(job.model_type))) return null;
  const summaries = selectedJobs.map(summaryFor);
  return { summaries, anyStacking: summaries.some(s => s?.isStacking) };
}

export function EnsembleRows({ selectedJobs, ensembleData }: {
  selectedJobs: JobInfo[];
  ensembleData: ReturnType<typeof prepareEnsembleData>;
}) {
  if (!ensembleData) return null;
  return (
    <>
      <tr className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
        <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100">Base Models</td>
        {ensembleData.summaries.map((s, ci) => (
          <td key={selectedJobs[ci]?.job_id ?? ci} className="px-4 py-2 text-gray-600 dark:text-gray-300">
            {s ? (s.baseEstimators.length > 0 ? s.baseEstimators.map(formatBaseEstimator).join(', ') : '—') : '—'}
          </td>
        ))}
      </tr>
      {ensembleData.anyStacking && (
        <tr className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
          <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100">Final Estimator</td>
          {ensembleData.summaries.map((s, ci) => (
            <td key={selectedJobs[ci]?.job_id ?? ci} className="px-4 py-2 text-gray-600 dark:text-gray-300">
              {s?.isStacking ? (s.finalEstimator ? formatBaseEstimator(s.finalEstimator) : '—') : '—'}
            </td>
          ))}
        </tr>
      )}
    </>
  );
}
