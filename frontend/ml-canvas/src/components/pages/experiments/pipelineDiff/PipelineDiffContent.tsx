import { ReactFlow, Background, Controls, ReactFlowProvider } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ArrowLeftRight } from 'lucide-react';
import { shortRunId } from '../../ExperimentsPage/utils/jobMeta';
import { uniqueNodeDiffs, type GraphDiff, type NodeDiff } from '../../../../core/utils/graphDiff';
import { DiffNode } from '../DiffNode';
import { StatusDot } from '../StatusDot';
import type { JobLite, SideGraph } from '../pipelineDiffLayout';
import { describeJob, formatTimestamp } from './jobDescription';

const nodeTypes = { diff: DiffNode };

interface ContentProps {
  baselineJob: JobLite;
  candidateJob: JobLite;
  diff: GraphDiff;
  styled: { baseline: SideGraph; candidate: SideGraph };
  onSwap: () => void;
}

function PipelineDiffSummary({ baselineJob, candidateJob, summary, modifiedNodes, onSwap }: {
  baselineJob: JobLite;
  candidateJob: JobLite;
  summary: GraphDiff['summary'];
  modifiedNodes: NodeDiff[];
  onSwap: () => void;
}) {
  return (
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
          onClick={onSwap}
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
  );
}

function PipelineDiffGraph({ comparisonRole, job, sideGraph }: {
  comparisonRole: 'Baseline' | 'Candidate';
  job: JobLite;
  sideGraph: SideGraph;
}) {
  return (
    <div
      className="rounded-md border bg-card overflow-hidden flex flex-col"
      style={{ height: 320 }}
    >
      <div className="px-3 py-2 border-b text-xs flex items-start justify-between gap-3 bg-muted/30">
        <div className="min-w-0 truncate">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold">{comparisonRole}</span>
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
}

function PipelineDiffChanges({ modifiedNodes }: { modifiedNodes: NodeDiff[] }) {
  if (modifiedNodes.length === 0) return null;
  return (
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
  );
}

export function PipelineDiffContent({ baselineJob, candidateJob, diff, styled, onSwap }: ContentProps) {
  // Collapse entries stored under both the baseline and candidate id so
  // renamed-and-modified nodes appear only once in the change list.
  const modifiedNodes = uniqueNodeDiffs(diff.nodes).filter((node) => node.status !== 'unchanged');
  return (
    <div className="space-y-4">
      <PipelineDiffSummary baselineJob={baselineJob} candidateJob={candidateJob}
        summary={diff.summary} modifiedNodes={modifiedNodes} onSwap={onSwap} />
      <div className="flex flex-col gap-4">
        <PipelineDiffGraph comparisonRole="Baseline" job={baselineJob} sideGraph={styled.baseline} />
        <PipelineDiffGraph comparisonRole="Candidate" job={candidateJob} sideGraph={styled.candidate} />
      </div>
      <PipelineDiffChanges modifiedNodes={modifiedNodes} />
    </div>
  );
}
