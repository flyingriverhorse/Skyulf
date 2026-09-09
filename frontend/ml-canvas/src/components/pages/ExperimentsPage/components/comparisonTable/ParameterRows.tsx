import type { JobInfo } from '../../../../../core/api/jobs';
import { getHyperparamDescription } from '../../../../../core/utils/format';
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import { getModelParams } from './config';

export function ParameterRows({ selectedJobs, paramsAllKeys }: { selectedJobs: JobInfo[]; paramsAllKeys: string[] }) {
  return <>{
    paramsAllKeys.length === 0 ? (
      <tr className="bg-white dark:bg-gray-800">
        <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8 italic" colSpan={selectedJobs.length + 1}>
          Default parameters (none customized)
        </td>
      </tr>
    ) : paramsAllKeys.map(paramKey => (
      <tr key={paramKey} className="bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700/50">
        <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">
          <div className="flex items-center gap-1">
            {paramKey}
            {getHyperparamDescription(paramKey) && <InfoTooltip size="sm" text={getHyperparamDescription(paramKey)!} />}
          </div>
        </td>
        {selectedJobs.map(job => {
          const params = getModelParams(job);
          const val = params[paramKey];
          return (
            <td key={job.job_id} className="px-4 py-1.5 font-mono text-gray-600 dark:text-gray-300">
              {val === undefined ? '-' : typeof val === 'object' ? JSON.stringify(val) : String(val)}
            </td>
          );
        })}
      </tr>
    ))}</>;
}
