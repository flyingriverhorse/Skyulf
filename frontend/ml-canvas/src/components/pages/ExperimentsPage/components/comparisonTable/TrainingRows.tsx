import type { JobInfo } from '../../../../../core/api/jobs';
import { getTrainingConfigDescription } from '../../../../../core/utils/format';
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import { hasTuningMetadata } from '../../utils/jobMeta';
import { getJobConfig } from './config';
import { trainingConfigValue } from './trainingConfig';

export function TrainingRows({ selectedJobs }: { selectedJobs: JobInfo[] }) {
  const fields = [
    'Target Column', 'CV Enabled', 'CV Method', 'CV Folds', 'CV Shuffle', 'CV Random State',
    ...(selectedJobs.some(j => hasTuningMetadata(j)) ? ['Strategy', 'Strategy Params', 'Metric', 'Trials'] : []),
  ];
  return (
    <>
      {fields.map(field => (
        <tr key={field} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
            <div className="flex items-center gap-1">
              {field}
              {getTrainingConfigDescription(field) && <InfoTooltip size="sm" text={getTrainingConfigDescription(field)!} />}
            </div>
          </td>
          {selectedJobs.map(job => {
            const cfg = getJobConfig(job);

            if (!cfg) {
              return <td key={job.job_id} className="px-4 py-1.5 text-gray-400">-</td>;
            }

            const value = trainingConfigValue(field, job, cfg);

            return (
              <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300 capitalize">
                {value}
              </td>
            );
          })}
        </tr>
      ))}
    </>
  );
}
