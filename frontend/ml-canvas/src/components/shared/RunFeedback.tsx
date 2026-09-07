import { useJobStore } from '../../core/store/useJobStore';
import type { SubmittedRun } from '../../core/types/runFeedback';
import type { TaskType } from '../../core/types/taskType';

/** Describe a submitted run using its own job IDs, including partial or missing results. */
export function RunFeedback({ run, task, onOpen }: { run: SubmittedRun; task?: TaskType; onOpen?: () => void }) {
  const jobs = useJobStore(state => state.jobs);
  const runJobs = useJobStore(state => state.runJobs);
  const toggleDrawer = useJobStore(state => state.toggleDrawer);
  const counts = new Map<string, number>();
  for (const id of new Set(run.jobIds)) {
    const status = (runJobs[id] ?? jobs.find(job => job.job_id === id))?.status;
    const label = status === 'succeeded' ? 'completed' : status ?? 'awaiting status';
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const summary = counts.size === 0 || (counts.size === 1 && counts.has('awaiting status'))
    ? 'Awaiting status'
    : [...counts].map(([status, count]) => `${count} ${status}`).join(' · ');
  return <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-xs">
    <p role="status" aria-atomic="true" className="min-w-0 break-words [overflow-wrap:anywhere] text-muted-foreground">
      {run.label}: {summary}
    </p>
    <button type="button" onClick={() => {
      onOpen?.();
      const store = useJobStore.getState();
      store.setInspectedRun(task ? null : run);
      if (task) store.setTab(task);
      toggleDrawer(true);
    }} className="shrink-0 rounded text-primary underline underline-offset-2 focus-ring">
      View jobs
    </button>
  </div>;
}
