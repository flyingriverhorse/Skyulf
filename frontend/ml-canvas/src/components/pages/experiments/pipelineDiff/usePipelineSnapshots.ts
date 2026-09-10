import { useEffect, useState } from 'react';
import { jobsApi } from '../../../../core/api/jobs';
import { readSideFromGraph, type JobLite, type SideGraph } from '../pipelineDiffLayout';

export type SnapshotState =
  | { status: 'loading' }
  | { status: 'ready'; graph: SideGraph }
  | { status: 'missing' }
  | { status: 'error'; message: string };

function readSnapshot(result: PromiseSettledResult<Awaited<ReturnType<typeof jobsApi.getJob>>>): SnapshotState {
  if (result.status === 'fulfilled') {
    if (!result.value.graph) return { status: 'missing' };
    return { status: 'ready', graph: readSideFromGraph(result.value.graph) };
  }
  return {
    status: 'error',
    message: result.reason instanceof Error ? result.reason.message : 'Failed to load job graph',
  };
}

export function usePipelineSnapshots(selectedJobs: JobLite[]) {
  const [snapshots, setSnapshots] = useState<Record<string, SnapshotState>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (selectedJobs.length !== 2) {
      setSnapshots({});
      setLoading(false);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    const [firstJob, secondJob] = selectedJobs as [JobLite, JobLite];
    setSnapshots({
      [firstJob.job_id]: { status: 'loading' },
      [secondJob.job_id]: { status: 'loading' },
    });
    Promise.allSettled(selectedJobs.map(job => jobsApi.getJob(job.job_id)))
      .then((results) => {
        if (cancelled) return;
        const next: Record<string, SnapshotState> = {};
        results.forEach((result, index) => {
          const job = selectedJobs[index];
          if (!job) return;
          next[job.job_id] = readSnapshot(result);
        });
        setSnapshots(next);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load job graphs');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedJobs]);

  return { snapshots, loading, error };
}
