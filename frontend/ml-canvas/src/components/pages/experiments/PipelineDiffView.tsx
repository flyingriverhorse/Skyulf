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
import { Loader2, GitCompare } from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import {
  diffGraphs,
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

export const PipelineDiffView: React.FC<Props> = ({ jobs }) => {
  const [leftGraph, setLeftGraph] = useState<SideGraph | null>(null);
  const [rightGraph, setRightGraph] = useState<SideGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Always diff in display order (sidebar's selection order). Two
  // jobs is the supported case; we surface a guard message otherwise.
  const [leftJob, rightJob] = jobs;

  useEffect(() => {
    if (!leftJob || !rightJob) {
      setLeftGraph(null);
      setRightGraph(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([jobsApi.getJob(leftJob.job_id), jobsApi.getJob(rightJob.job_id)])
      .then(([l, r]) => {
        if (cancelled) return;
        setLeftGraph(readSideFromGraph(l.graph));
        setRightGraph(readSideFromGraph(r.graph));
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
  }, [leftJob, rightJob]);

  const diff = useMemo<GraphDiff | null>(() => {
    if (!leftGraph || !rightGraph) return null;
    return diffGraphs(leftGraph.nodes, leftGraph.edges, rightGraph.nodes, rightGraph.edges);
  }, [leftGraph, rightGraph]);

  const styled = useMemo(() => {
    if (!leftGraph || !rightGraph || !diff) return null;
    const { positions } = layoutUnified(leftGraph, rightGraph, diff.aliases);
    return {
      left: applyLayout(applyDiffStylingToSide(leftGraph, diff, 'left'), positions, diff.aliases),
      right: applyLayout(applyDiffStylingToSide(rightGraph, diff, 'right'), positions, diff.aliases),
    };
  }, [leftGraph, rightGraph, diff]);

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

  if (error) {
    return (
      <div className="rounded-md border border-red-500/40 bg-red-500/5 p-4 text-sm text-red-600 dark:text-red-400">
        Failed to load pipeline graphs: {error}
      </div>
    );
  }

  if (!styled || !diff || !leftJob || !rightJob) return null;

  const summary = diff.summary;
  const modifiedNodes = Array.from(diff.nodes.values()).filter(
    (n: NodeDiff) => n.status !== 'unchanged',
  );

  return (
    <div className="space-y-4">
      <div className="rounded-md border bg-card p-4">
        <div className="flex flex-wrap items-center gap-3 text-sm">
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
        {modifiedNodes.length === 0 && (
          <p className="mt-2 text-xs text-muted-foreground">
            No structural or config differences detected — the two pipelines are functionally
            identical (only run-time results differ).
          </p>
        )}
      </div>

      {/* Stacked top/bottom (Baseline above Candidate) so each canvas
          gets the full container width — linear pipelines fit on a
          single row instead of squeezing into half-width side-by-side
          panels. The unified layout still keeps unchanged nodes at
          the same X coordinate so the eye scans straight up/down. */}
      <div className="flex flex-col gap-4">
        {(['left', 'right'] as const).map((side) => {
          const job = side === 'left' ? leftJob : rightJob;
          const sideGraph = side === 'left' ? styled.left : styled.right;
          return (
            <div
              key={side}
              className="rounded-md border bg-card overflow-hidden flex flex-col"
              style={{ height: 320 }}
            >
              <div className="px-3 py-2 border-b text-xs flex items-center justify-between bg-muted/30">
                <div className="truncate">
                  <span className="font-semibold mr-2">
                    {side === 'left' ? 'Baseline' : 'Candidate'}
                  </span>
                  <span className="text-muted-foreground">
                    {job.model_type ?? job.job_id.slice(0, 8)}
                  </span>
                </div>
                <span className="text-[10px] text-muted-foreground font-mono">
                  {job.job_id.slice(0, 8)}
                </span>
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
