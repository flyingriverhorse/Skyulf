import type React from 'react';
import { Rocket, Trophy } from 'lucide-react';
import type { JobInfo } from '../../../../../core/api/jobs';
import type { JobListSidebar } from '../JobListSidebar';
type SidebarProps = React.ComponentProps<typeof JobListSidebar>;

/** Completed model runs expose their existing winner and deployment actions. */
export function JobActions({ job, handlePromote, handleDeploy }: Pick<SidebarProps, 'handlePromote' | 'handleDeploy'> & { job: JobInfo }) {
  return <>
    {job.status === 'completed' && (job.job_type === 'training' || job.job_type === 'tuning') && (
      <>
        <button
          onClick={(e) => { void handlePromote(e, job); }}
          className={`p-1 rounded transition-colors ${job.promoted_at
              ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400'
              : 'hover:bg-amber-100 dark:hover:bg-amber-900/20 text-gray-500 dark:text-gray-400'
            }`}
          title={job.promoted_at ? 'Unpromote' : 'Promote as Winner'}
        >
          <Trophy className="w-3 h-3" />
        </button>
        <button
          onClick={(e) => { void handleDeploy(e, job.job_id); }}
          className="p-1 hover:bg-blue-100 dark:hover:bg-blue-900 rounded text-blue-600 dark:text-blue-400"
          title="Deploy to Test"
        >
          <Rocket className="w-3 h-3" />
        </button>
      </>
    )}</>;
}
