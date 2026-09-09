import { GitCompare, Loader2 } from 'lucide-react';
import type { JobLite } from '../pipelineDiffLayout';
import type { SnapshotState } from './usePipelineSnapshots';
import { snapshotMessage } from './jobDescription';

export function PipelineDiffSelection({ count }: { count: number }) {
  return (
    <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground flex items-start gap-3">
      <GitCompare className="w-5 h-5 mt-0.5 shrink-0" />
      <div>
        <div className="font-medium text-foreground mb-1">Pick exactly two runs</div>
        <p>
          The Pipeline Diff view compares two pipeline graphs side by side and color-codes the
          nodes / edges that changed. Select two runs in the sidebar to enable it
          ({count} selected).
        </p>
      </div>
    </div>
  );
}

export function PipelineDiffLoading() {
  return (
    <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground flex items-center gap-2">
      <Loader2 className="w-4 h-4 animate-spin" />
      Loading pipeline graphs…
    </div>
  );
}

function describeSnapshotIssue(role: 'Baseline' | 'Candidate', job: JobLite, snapshot: SnapshotState | undefined) {
  if (snapshot?.status === 'missing') {
    return snapshotMessage(role, job, 'has no saved pipeline snapshot');
  }
  if (snapshot?.status === 'error') {
    return snapshotMessage(role, job, `could not be loaded: ${snapshot.message}`);
  }
  return null;
}

export function PipelineDiffSnapshotIssues({ baselineJob, candidateJob, snapshots }: {
  baselineJob: JobLite;
  candidateJob: JobLite;
  snapshots: Record<string, SnapshotState>;
}) {
  const baselineMessage = describeSnapshotIssue('Baseline', baselineJob, snapshots[baselineJob.job_id]);
  const candidateMessage = describeSnapshotIssue('Candidate', candidateJob, snapshots[candidateJob.job_id]);
  return (
    <div className="space-y-3 rounded-md border bg-card p-4 text-sm text-muted-foreground">
      <div className="flex items-start gap-3 text-foreground">
        <GitCompare className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <div className="font-medium">Pipeline Diff needs two saved snapshots</div>
          <p className="text-muted-foreground">
            Selection order sets the baseline first and the candidate second; use Swap to flip the roles.
          </p>
        </div>
      </div>
      {baselineMessage && <p>{baselineMessage}</p>}
      {candidateMessage && <p>{candidateMessage}</p>}
    </div>
  );
}
