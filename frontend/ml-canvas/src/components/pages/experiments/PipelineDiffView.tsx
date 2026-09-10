// L5: side-by-side visual diff of two historical pipeline snapshots.
import React, { useEffect, useMemo, useState } from 'react';
import type { JobLite } from './pipelineDiffLayout';
import { PipelineDiffContent } from './pipelineDiff/PipelineDiffContent';
import { PipelineDiffLoading, PipelineDiffSelection, PipelineDiffSnapshotIssues } from './pipelineDiff/PipelineDiffStates';
import { describeJob } from './pipelineDiff/jobDescription';
import { usePipelineComparison } from './pipelineDiff/usePipelineComparison';
import { usePipelineSnapshots, type SnapshotState } from './pipelineDiff/usePipelineSnapshots';

interface Props {
  jobs: JobLite[];
}

type ResultProps = ReturnType<typeof usePipelineComparison> & {
  baselineJob: JobLite;
  candidateJob: JobLite;
  snapshots: Record<string, SnapshotState>;
  error: string | null;
  onSwap: () => void;
};

function PipelineDiffResult({ baselineJob, candidateJob, snapshots, error, diff, styled, onSwap }: ResultProps) {
  const snapshotIssue = [snapshots[baselineJob.job_id], snapshots[candidateJob.job_id]].find(
    (snapshot) => snapshot && snapshot.status !== 'ready',
  );
  if (snapshotIssue) {
    return <PipelineDiffSnapshotIssues baselineJob={baselineJob} candidateJob={candidateJob} snapshots={snapshots} />;
  }
  if (error) {
    return (
      <div className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-600 dark:text-red-400">
        Failed to load pipeline graphs for {describeJob(baselineJob)} and{' '}
        {describeJob(candidateJob)}: {error}. Re-run the comparison or refresh the page.
      </div>
    );
  }
  if (!styled || !diff) return null;
  return <PipelineDiffContent baselineJob={baselineJob} candidateJob={candidateJob}
    styled={styled} diff={diff} onSwap={onSwap} />;
}

export const PipelineDiffView: React.FC<Props> = ({ jobs }) => {
  const [swapped, setSwapped] = useState(false);
  const selectedJobs = useMemo(
    () => (jobs.length === 2 ? [jobs[0]!, jobs[1]!] : []),
    [jobs],
  );
  useEffect(() => {
    setSwapped(false);
  }, [selectedJobs]);

  const { snapshots, loading, error } = usePipelineSnapshots(selectedJobs);
  const comparison = usePipelineComparison(selectedJobs, swapped, snapshots);
  const { baselineJob, candidateJob } = comparison;

  if (jobs.length !== 2) return <PipelineDiffSelection count={jobs.length} />;
  if (loading) return <PipelineDiffLoading />;
  if (!baselineJob || !candidateJob) return null;

  return <PipelineDiffResult {...comparison} baselineJob={baselineJob} candidateJob={candidateJob}
    snapshots={snapshots} error={error} onSwap={() => setSwapped((current) => !current)} />;
};
