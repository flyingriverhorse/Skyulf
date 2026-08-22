// L5: side-by-side visual diff of two pipeline graphs.
//
// Shown in the Experiments → Pipeline Diff tab when exactly two jobs
// are selected. Fetches each job's saved graph, runs `diffGraphs`,
// and renders two read-only React Flow viewers with diff-aware
// node/edge styling plus a change-list panel.
//
// Why a custom mini-node instead of CustomNodeWrapper:
//  - CustomNodeWrapper pulls live job state, perf overlay state, etc.
//    None of that is meaningful for a historical snapshot.
//  - We want the visual to focus on the diff (rings + labels), not
//    the editor's chrome.

import React, { useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ArrowLeftRight, GitCompare, Loader2 } from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import { shortRunId } from '../ExperimentsPage/utils/jobMeta';
import {
  diffGraphs,
  uniqueNodeDiffs,
  type GraphDiff,
  type NodeDiff,
} from '../../../core/utils/graphDiff';
import { DiffNode } from './DiffNode';
import { StatusDot } from './StatusDot';
import {
  applyDiffStylingToSide,
  applyLayout,
  layoutUnified,
  readSideFromGraph,
  type JobLite,
  type SideGraph,
} from './pipelineDiffLayout';

interface Props {
  jobs: JobLite[];
}

const nodeTypes = { diff: DiffNode };

type SnapshotState =
  | { status: 'loading' }
  | { status: 'ready'; graph: SideGraph }
  | { status: 'missing' }
  | { status: 'error'; message: string };

const formatTimestamp = (timestamp?: string): string => {
  if (!timestamp) return 'Unknown time';
  const value = new Date(timestamp);
  return Number.isNaN(value.getTime()) ? timestamp : value.toLocaleString();
};

const describeJob = (job: JobLite): string => {
  const dataset = job.dataset_name ?? 'Unknown dataset';
  const model = job.model_type ?? 'Unknown model';
  return `${dataset} · ${model} · ${formatTimestamp(job.created_at)}`;
};

const snapshotMessage = (role: 'Baseline' | 'Candidate', job: JobLite, detail: string): string =>
  `${role} run ${shortRunId(job)} (${describeJob(job)}) ${detail}. Re-run the pipeline or save the canvas snapshot, then compare again.`;

export const PipelineDiffView: React.FC<Props> = ({ jobs }) => {
  const [swapped, setSwapped] = useState(false);
  const [snapshots, setSnapshots] = useState<Record<string, SnapshotState>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedJobs = useMemo(
    () => (jobs.length === 2 ? [jobs[0]!, jobs[1]!] : []),
    [jobs],
  );
  const baselineJob = selectedJobs.length === 2 ? (swapped ? selectedJobs[1] : selectedJobs[0]) : undefined;
  const candidateJob = selectedJobs.length === 2 ? (swapped ? selectedJobs[0] : selectedJobs[1]) : undefined;

  useEffect(() => {
    setSwapped(false);
  }, [selectedJobs]);

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
          if (result.status === 'fulfilled') {
            if (!result.value.graph) {
              next[job.job_id] = { status: 'missing' };
              return;
            }
            next[job.job_id] = { status: 'ready', graph: readSideFromGraph(result.value.graph) };
            return;
          }
          next[job.job_id] = {
            status: 'error',
            message: result.reason instanceof Error ? result.reason.message : 'Failed to load job graph',
          };
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

  const diff = useMemo<GraphDiff | null>(() => {
    const baseline = baselineJob ? snapshots[baselineJob.job_id] : undefined;
    const candidate = candidateJob ? snapshots[candidateJob.job_id] : undefined;
    if (baseline?.status !== 'ready' || candidate?.status !== 'ready') return null;
    return diffGraphs(baseline.graph.nodes, baseline.graph.edges, candidate.graph.nodes, candidate.graph.edges);
  }, [baselineJob, candidateJob, snapshots]);

  const styled = useMemo(() => {
    const baseline = baselineJob ? snapshots[baselineJob.job_id] : undefined;
    const candidate = candidateJob ? snapshots[candidateJob.job_id] : undefined;
    if (baseline?.status !== 'ready' || candidate?.status !== 'ready' || !diff) return null;
    const { positions } = layoutUnified(baseline.graph, candidate.graph, diff.aliases);
    return {
      baseline: applyLayout(applyDiffStylingToSide(baseline.graph, diff, 'left'), positions, diff.aliases),
      candidate: applyLayout(applyDiffStylingToSide(candidate.graph, diff, 'right'), positions, diff.aliases),
    };
  }, [baselineJob, candidateJob, snapshots, diff]);

  if (jobs.length !== 2) {
    return (
      <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground flex items-start gap-3">
        <GitCompare className="w-5 h-5 mt-0.5 shrink-0" />
        <div>
          <div className="font-medium text-foreground mb-1">Pick exactly two runs</div>
          <p>
            The Pipeline Diff view compares two pipeline graphs side by side and color-codes the
            nodes / edges that changed. Select two runs in the sidebar to enable it
            ({jobs.length} selected).
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="rounded-md border bg-card p-6 text-sm text-muted-foreground flex items-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        Loading pipeline graphs…
      </div>
    );
  }

  const baselineSnapshot = baselineJob ? snapshots[baselineJob.job_id] : undefined;
  const candidateSnapshot = candidateJob ? snapshots[candidateJob.job_id] : undefined;
  const snapshotIssue = [baselineSnapshot, candidateSnapshot].find(
    (snapshot) => snapshot && snapshot.status !== 'ready',
  );
  if (snapshotIssue && baselineJob && candidateJob) {
    const baselineMessage =
      baselineSnapshot?.status === 'missing'
        ? snapshotMessage('Baseline', baselineJob, 'has no saved pipeline snapshot')
        : baselineSnapshot?.status === 'error'
          ? snapshotMessage('Baseline', baselineJob, `could not be loaded: ${baselineSnapshot.message}`)
          : null;
    const candidateMessage =
      candidateSnapshot?.status === 'missing'
        ? snapshotMessage('Candidate', candidateJob, 'has no saved pipeline snapshot')
        : candidateSnapshot?.status === 'error'
          ? snapshotMessage('Candidate', candidateJob, `could not be loaded: ${candidateSnapshot.message}`)
          : null;
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

  if (error) {
    return (
      <div className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-600 dark:text-red-400">
        Failed to load pipeline graphs for {baselineJob ? describeJob(baselineJob) : 'baseline'} and{' '}
        {candidateJob ? describeJob(candidateJob) : 'candidate'}: {error}. Re-run the comparison or refresh the page.
      </div>
    );
  }

  if (!styled || !diff || !baselineJob || !candidateJob) return null;

  const summary = diff.summary;
  // uniqueNodeDiffs collapses the double-registration (same NodeDiff
  // stored under both the baseline and candidate id) before rendering,
  // otherwise every renamed-and-modified node lists twice.
  const modifiedNodes = uniqueNodeDiffs(diff.nodes).filter(
    (n: NodeDiff) => n.status !== 'unchanged',
  );

  return (
    <div className="space-y-4">
      <div className="rounded-md border bg-card p-4">
        <div className="flex flex-wrap items-start justify-between gap-3 text-sm">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-medium">Diff summary:</span>
            {summary.nodesAdded > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <StatusDot status="added" /> {summary.nodesAdded} added
              </span>
            )}
            {summary.nodesRemoved > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <StatusDot status="removed" /> {summary.nodesRemoved} removed
              </span>
            )}
            {summary.nodesModified > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <StatusDot status="modified" /> {summary.nodesModified} modified
              </span>
            )}
            <span className="text-muted-foreground">
              {summary.nodesUnchanged} unchanged
              {summary.nodesRenamed > 0 && ` (${summary.nodesRenamed} renamed across runs)`}
              {(summary.edgesAdded > 0 || summary.edgesRemoved > 0) && (
                <> · edges {summary.edgesAdded}+ / {summary.edgesRemoved}−</>
              )}
            </span>
          </div>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium text-foreground hover:bg-muted"
            onClick={() => setSwapped((current) => !current)}
          >
            <ArrowLeftRight className="h-3.5 w-3.5" />
            Swap
          </button>
        </div>
        {modifiedNodes.length === 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            No structural or config differences detected between {describeJob(baselineJob)} and{' '}
            {describeJob(candidateJob)}. Swap the roles or re-run after saving a changed pipeline
            if you expected a different result.
          </p>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          Baseline uses the first selected run and Candidate uses the second selected run unless you
          swap them.
        </p>
      </div>

      <div className="flex flex-col gap-4">
        {([
          ['Baseline', baselineJob, styled.baseline],
          ['Candidate', candidateJob, styled.candidate],
        ] as const).map(([role, job, sideGraph]) => {
          return (
            <div
              key={role}
              className="rounded-md border bg-card overflow-hidden flex flex-col"
              style={{ height: 320 }}
            >
              <div className="px-3 py-2 border-b text-xs flex items-start justify-between gap-3 bg-muted/30">
                <div className="min-w-0 truncate">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold">{role}</span>
                    <span className="text-muted-foreground font-mono">{shortRunId(job)}</span>
                  </div>
                  <div className="text-muted-foreground">
                    {job.dataset_name ?? 'Unknown dataset'} · {job.model_type ?? 'Unknown model'} ·{' '}
                    {formatTimestamp(job.created_at)}
                  </div>
                </div>
              </div>
              <div className="flex-1 min-h-0">
                <ReactFlowProvider>
                  <ReactFlow
                    nodes={sideGraph.nodes}
                    edges={sideGraph.edges}
                    nodeTypes={nodeTypes}
                    fitView
                    fitViewOptions={{ padding: 0.2, includeHiddenNodes: false }}
                    minZoom={0.2}
                    maxZoom={1.5}
                    nodesDraggable={false}
                    nodesConnectable={false}
                    elementsSelectable={false}
                    // Page scroll must pass through this canvas.
                    // Without these flags the wheel either zooms the
                    // graph or pans it, which makes the diff section
                    // feel "sticky" when the user is scrolling the
                    // Experiments page.
                    panOnScroll={false}
                    zoomOnScroll={false}
                    zoomOnPinch={false}
                    zoomOnDoubleClick={false}
                    preventScrolling={false}
                    proOptions={{ hideAttribution: true }}
                  >
                    <Background gap={16} size={1} />
                    <Controls showInteractive={false} />
                  </ReactFlow>
                </ReactFlowProvider>
              </div>
            </div>
          );
        })}
      </div>

      {modifiedNodes.length > 0 && (
        <div className="rounded-md border bg-card">
          <div className="px-4 py-2 border-b font-medium text-sm">Changes</div>
          <ul className="divide-y">
            {modifiedNodes.map((n: NodeDiff) => (
              <li key={n.id} className="px-4 py-2 text-sm">
                <div className="flex items-center gap-2">
                  <StatusDot status={n.status} />
                  <span className="font-medium">{n.label}</span>
                  <span className="text-[10px] text-muted-foreground font-mono">{n.id}</span>
                  <span className="ml-auto text-xs text-muted-foreground capitalize">
                    {n.status}
                  </span>
                </div>
                {n.changeDescriptions.length > 0 && (
                  <ul className="mt-1 ml-4 text-xs text-muted-foreground space-y-0.5">
                    {n.changeDescriptions.map((d: string, i: number) => (
                      <li key={i} className="font-mono">
                        {d}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
