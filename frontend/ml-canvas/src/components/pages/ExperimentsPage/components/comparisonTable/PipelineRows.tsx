import type { JobInfo } from '../../../../../core/api/jobs';
import type { PipelineData } from './types';

export function PipelineRows({ selectedJobs, pipelineData }: { selectedJobs: JobInfo[]; pipelineData: PipelineData }) {
  return <>{
    pipelineData.hasSteps ? pipelineData.rows.map(({ nid, idx, cells, allSame }) => {
      // Shared trunk row → muted background; divergent
      // row (some columns dash, others differ) →
      // amber accent so the diff jumps out.
      const rowTone = allSame
        ? 'bg-white dark:bg-gray-800'
        : 'bg-amber-50/40 dark:bg-amber-900/10';
      return (
        <tr key={`pipeline-step-${nid}`} className={`${rowTone} hover:bg-gray-50 dark:hover:bg-gray-700/50`}>
          <td className="px-4 py-1.5 text-gray-500 dark:text-gray-400 pl-8">Step {idx + 1}</td>
          {cells.map((text, ci) => (
            <td
              key={selectedJobs[ci]?.job_id ?? ci}
              className={
                text === null
                  ? 'px-4 py-1.5 text-gray-300 dark:text-gray-600'
                  : allSame
                    ? 'px-4 py-1.5 text-gray-500 dark:text-gray-400'
                    : 'px-4 py-1.5 text-gray-900 dark:text-gray-100 font-medium'
              }
            >
              {text ?? <span className="text-gray-400">—</span>}
            </td>
          ))}
        </tr>
      );
    }) : (
      <tr className="bg-white dark:bg-gray-800">
        <td className="px-4 py-1.5 text-gray-400 italic pl-8" colSpan={selectedJobs.length + 1}>
          No upstream pipeline steps captured for these runs.
        </td>
      </tr>
    )}</>;
}
