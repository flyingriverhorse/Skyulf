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
  Handle,
  Position,
  MarkerType,
  ReactFlowProvider,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Loader2, GitCompare } from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import {
  diffGraphs,
  edgeDiffStroke,
  nodeDiffRingClass,
  type GraphDiff,
  type NodeDiff,
  type NodeDiffStatus,
} from '../../../core/utils/graphDiff';

interface JobLite {
  job_id: string;
  pipeline_id: string;
  model_type?: string;
  parent_pipeline_id?: string | null;
}

interface Props {
  jobs: JobLite[];
}

// Minimal read-only node shown in the side-by-side viewers. The diff
// status is encoded as a ring colour plus a tiny status pill so users
// can read the canvas at a glance. Explicit `Handle` elements are
// required on a custom React Flow node type — without them, edges
// fall back to (0,0) anchor points and visually detach from the node.
const NODE_W = 180;
const NODE_H = 64;

const DiffNode: React.FC<{
  data: { label: string; subLabel?: string; diffStatus: NodeDiffStatus };
}> = ({ data }) => {
  const ring = nodeDiffRingClass(data.diffStatus);
  const pill =
    data.diffStatus === 'added'
      ? { text: 'NEW', cls: 'bg-green-500/15 text-green-600 dark:text-green-400' }
      : data.diffStatus === 'removed'
      ? { text: 'GONE', cls: 'bg-red-500/15 text-red-600 dark:text-red-400' }
      : data.diffStatus === 'modified'
      ? { text: 'EDIT', cls: 'bg-amber-500/15 text-amber-600 dark:text-amber-400' }
      : null;
  return (
    <div
      className={`px-3 py-2 rounded-md border bg-card text-card-foreground shadow-sm flex flex-col justify-center ${ring}`}
      style={{ width: NODE_W, height: NODE_H }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#94a3b8', width: 6, height: 6, border: 'none' }}
      />
      <div className="flex items-center justify-between gap-2 min-w-0">
        <div className="text-xs font-medium truncate">{data.label}</div>
        {pill && (
          <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold shrink-0 ${pill.cls}`}>
            {pill.text}
          </span>
        )}
      </div>
      {data.subLabel && (
        <div className="text-[10px] text-muted-foreground truncate font-mono">{data.subLabel}</div>
      )}
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#94a3b8', width: 6, height: 6, border: 'none' }}
      />
    </div>
  );
};

const nodeTypes = { diff: DiffNode };

interface SideGraph {
  nodes: Node[];
  edges: Edge[];
}

function applyDiffStylingToSide(
  raw: SideGraph,
  diff: GraphDiff,
  side: 'left' | 'right',
): SideGraph {
  const styledNodes: Node[] = raw.nodes.map((n) => {
    const d = diff.nodes.get(n.id);
    // On the left (baseline), `added` nodes don't exist; on the
    // right (candidate), `removed` nodes don't exist. The other
    // status flows through unchanged.
    const status: NodeDiffStatus =
      !d ? 'unchanged'
      : d.status === 'added' && side === 'left' ? 'unchanged'
      : d.status === 'removed' && side === 'right' ? 'unchanged'
      : d.status;
    const data = (n.data ?? {}) as Record<string, unknown>;
    // Prefer the human-friendly step type (e.g. "basic_training",
    // "outlier_handling") as the primary label and demote the raw
    // node id to a smaller monospace sub-line. This avoids the
    // "label vs id collision" the user flagged where both lines
    // showed the same uuid-ish string.
    const stepType =
      typeof data.definitionType === 'string' ? (data.definitionType as string) : undefined;
    const explicitLabel =
      typeof data.label === 'string' && data.label !== n.id ? (data.label as string) : undefined;
    const label = explicitLabel ?? stepType ?? n.id;
    const subLabel = label === n.id ? undefined : n.id;
    return {
      ...n,
      type: 'diff',
      draggable: false,
      selectable: false,
      data: { label, subLabel, diffStatus: status },
    };
  });
  const styledEdges: Edge[] = raw.edges.map((e) => {
    const d = diff.edges.get(e.id);
    const status =
      !d ? 'unchanged'
      : d.status === 'added' && side === 'left' ? 'unchanged'
      : d.status === 'removed' && side === 'right' ? 'unchanged'
      : d.status;
    const stroke = edgeDiffStroke(status) ?? '#94a3b8';
    return {
      ...e,
      type: 'smoothstep',
      animated: false,
      style: { stroke, strokeWidth: stroke === '#94a3b8' ? 1.5 : 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: stroke, width: 16, height: 16 },
    };
  });
  return { nodes: styledNodes, edges: styledEdges };
}

function readSideFromGraph(graph: unknown): SideGraph {
  if (!graph || typeof graph !== 'object') return { nodes: [], edges: [] };
  const g = graph as { nodes?: unknown; edges?: unknown };
  const rawNodes = Array.isArray(g.nodes) ? g.nodes : [];
  const rawEdges = Array.isArray(g.edges) ? g.edges : [];

  // The backend stores two different graph shapes under JobInfo.graph:
  //   1. React Flow snapshot: { nodes: [{id, position, data, type}], edges: [{id, source, target}] }
  //      — used by /pipeline/save and the canvas autosave path.
  //   2. Engine config: { nodes: [{node_id, step_type, params, inputs}], metadata }
  //      — what the training/tuning API actually persists per job
  //      (see backend/ml_pipeline/api.py line ~370). No edges array,
  //      no positions, no `id` field.
  //
  // Detect the engine shape by the absence of `id` on the first node
  // and translate to a React-Flow-ish shape so the diff renders.
  const looksLikeEngineShape =
    rawNodes.length > 0 &&
    typeof (rawNodes[0] as Record<string, unknown>).id !== 'string' &&
    typeof (rawNodes[0] as Record<string, unknown>).node_id === 'string';

  if (looksLikeEngineShape) {
    return engineGraphToSide(rawNodes);
  }

  // Defensive normalisation for the React Flow snapshot. Older job
  // snapshots in the DB occasionally store nodes without a `position`
  // (or with `position: null`) which makes React Flow throw `Cannot
  // read properties of undefined (reading 'x')` on render. We
  // synthesise a fallback grid layout for any node missing a valid
  // `{x, y}` so the diff view always renders.
  const nodes: Node[] = [];
  rawNodes.forEach((raw, idx) => {
    if (!raw || typeof raw !== 'object') return;
    const r = raw as Record<string, unknown>;
    const id = typeof r.id === 'string' ? r.id : null;
    if (!id) return;
    const pos = r.position as { x?: unknown; y?: unknown } | null | undefined;
    const hasValidPos =
      pos && typeof pos === 'object' && typeof pos.x === 'number' && typeof pos.y === 'number';
    const fallback = { x: (idx % 4) * 220, y: Math.floor(idx / 4) * 140 };
    nodes.push({
      ...(r as object),
      id,
      position: hasValidPos ? { x: pos.x as number, y: pos.y as number } : fallback,
      data: (r.data as Record<string, unknown> | undefined) ?? {},
    } as Node);
  });

  const edges: Edge[] = [];
  rawEdges.forEach((raw, idx) => {
    if (!raw || typeof raw !== 'object') return;
    const r = raw as Record<string, unknown>;
    const source = typeof r.source === 'string' ? r.source : null;
    const target = typeof r.target === 'string' ? r.target : null;
    if (!source || !target) return;
    const id = typeof r.id === 'string' ? r.id : `e-${source}-${target}-${idx}`;
    edges.push({ ...(r as object), id, source, target } as Edge);
  });

  return { nodes, edges };
}

// Translate the engine-config graph (what the training/tuning API
// persists) into a React-Flow-ish SideGraph. Edges are derived from
// each node's `inputs` array. Positions are NOT assigned here — the
// view computes a unified layout across both sides so identical
// nodes line up between the baseline and candidate canvases.
function engineGraphToSide(rawNodes: unknown[]): SideGraph {
  type EngineNode = {
    node_id: string;
    step_type?: string | undefined;
    params?: Record<string, unknown> | undefined;
    inputs?: string[] | undefined;
  };
  const engineNodes: EngineNode[] = [];
  for (const raw of rawNodes) {
    if (!raw || typeof raw !== 'object') continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.node_id !== 'string') continue;
    engineNodes.push({
      node_id: r.node_id,
      step_type: typeof r.step_type === 'string' ? r.step_type : undefined,
      params:
        r.params && typeof r.params === 'object'
          ? (r.params as Record<string, unknown>)
          : undefined,
      inputs: Array.isArray(r.inputs) ? (r.inputs as string[]).filter((s) => typeof s === 'string') : [],
    });
  }

  const nodes: Node[] = engineNodes.map((n) => ({
    id: n.node_id,
    position: { x: 0, y: 0 }, // overwritten by layoutUnified
    data: {
      label: n.node_id,
      definitionType: n.step_type,
      // params land here so diffGraphs picks them up as the
      // user-meaningful config payload.
      ...(n.params ?? {}),
    },
  }));

  const edges: Edge[] = [];
  engineNodes.forEach((n) => {
    (n.inputs ?? []).forEach((src, i) => {
      edges.push({
        id: `e-${src}-${n.node_id}-${i}`,
        source: src,
        target: n.node_id,
      } as Edge);
    });
  });

  return { nodes, edges };
}

// Compute a shared layered layout from the union of nodes/edges of
// both sides. Each node's id maps to a single (col, row) — and thus
// (x, y) — that is identical across baseline and candidate canvases,
// so an "unchanged" node sits at the same screen position in both
// viewers and the human eye can scan vertically for differences.
const COL_GAP = 80;
const ROW_GAP = 36;

function layoutUnified(
  left: SideGraph,
  right: SideGraph,
): { positions: Map<string, { x: number; y: number }>; width: number; height: number } {
  const allNodeIds = new Set<string>();
  left.nodes.forEach((n) => allNodeIds.add(n.id));
  right.nodes.forEach((n) => allNodeIds.add(n.id));

  const inputs = new Map<string, Set<string>>();
  allNodeIds.forEach((id) => inputs.set(id, new Set()));
  [...left.edges, ...right.edges].forEach((e) => {
    if (allNodeIds.has(e.source) && allNodeIds.has(e.target)) {
      inputs.get(e.target)?.add(e.source);
    }
  });

  const depth = new Map<string, number>();
  const computeDepth = (id: string, stack: Set<string>): number => {
    if (depth.has(id)) return depth.get(id) as number;
    if (stack.has(id)) return 0;
    stack.add(id);
    const ins = Array.from(inputs.get(id) ?? []);
    const d = ins.length === 0 ? 0 : Math.max(...ins.map((i) => computeDepth(i, stack))) + 1;
    stack.delete(id);
    depth.set(id, d);
    return d;
  };
  allNodeIds.forEach((id) => computeDepth(id, new Set()));

  // Group ids by column. Sort within a column alphabetically for a
  // stable, deterministic layout (otherwise re-renders shuffle rows).
  const byCol = new Map<number, string[]>();
  allNodeIds.forEach((id) => {
    const d = depth.get(id) ?? 0;
    const col = byCol.get(d) ?? [];
    col.push(id);
    byCol.set(d, col);
  });
  byCol.forEach((ids) => ids.sort());

  const colCount = byCol.size === 0 ? 1 : Math.max(...byCol.keys()) + 1;
  const maxRows = Math.max(1, ...Array.from(byCol.values()).map((c) => c.length));

  const stepX = NODE_W + COL_GAP;
  const stepY = NODE_H + ROW_GAP;
  const totalH = maxRows * stepY;

  const positions = new Map<string, { x: number; y: number }>();
  byCol.forEach((ids, col) => {
    // Center each column vertically inside the canvas viewport — looks
    // tidier than top-aligning short columns next to tall ones.
    const colH = ids.length * stepY;
    const yOffset = (totalH - colH) / 2;
    ids.forEach((id, row) => {
      positions.set(id, { x: col * stepX, y: yOffset + row * stepY });
    });
  });

  return {
    positions,
    width: colCount * stepX,
    height: totalH,
  };
}

function applyLayout(side: SideGraph, positions: Map<string, { x: number; y: number }>): SideGraph {
  return {
    nodes: side.nodes.map((n) => ({ ...n, position: positions.get(n.id) ?? n.position })),
    edges: side.edges,
  };
}

const StatusDot: React.FC<{ status: NodeDiffStatus }> = ({ status }) => (
  <span
    className={`inline-block w-2 h-2 rounded-full ${
      status === 'added'
        ? 'bg-green-500'
        : status === 'removed'
        ? 'bg-red-500'
        : status === 'modified'
        ? 'bg-amber-500'
        : 'bg-slate-400'
    }`}
  />
);

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
    const { positions } = layoutUnified(leftGraph, rightGraph);
    return {
      left: applyLayout(applyDiffStylingToSide(leftGraph, diff, 'left'), positions),
      right: applyLayout(applyDiffStylingToSide(rightGraph, diff, 'right'), positions),
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
          <span className="inline-flex items-center gap-1.5">
            <StatusDot status="added" /> {summary.nodesAdded} added
          </span>
          <span className="inline-flex items-center gap-1.5">
            <StatusDot status="removed" /> {summary.nodesRemoved} removed
          </span>
          <span className="inline-flex items-center gap-1.5">
            <StatusDot status="modified" /> {summary.nodesModified} modified
          </span>
          <span className="text-muted-foreground">
            {summary.nodesUnchanged} unchanged · edges {summary.edgesAdded}+ /{' '}
            {summary.edgesRemoved}−
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
                    panOnScroll
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
