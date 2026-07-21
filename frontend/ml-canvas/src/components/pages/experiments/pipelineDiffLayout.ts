// Pure graph-shaping / layout helpers for the Pipeline Diff view.
//
// These translate a job's persisted graph (either a React Flow
// snapshot or an engine config) into a normalised `SideGraph`, apply
// diff-aware styling, and compute a unified layered layout shared by
// both the baseline and candidate canvases.

import { MarkerType, type Node, type Edge } from '@xyflow/react';
import {
  edgeDiffStroke,
  type GraphDiff,
  type NodeDiffStatus,
} from '../../../core/utils/graphDiff';

export interface JobLite {
  job_id: string;
  pipeline_id: string;
  model_type?: string;
  parent_pipeline_id?: string | null;
}

export interface SideGraph {
  nodes: Node[];
  edges: Edge[];
}

// Minimal read-only node shown in the side-by-side viewers. The diff
// status is encoded as a ring colour plus a tiny status pill so users
// can read the canvas at a glance. Explicit `Handle` elements are
// required on a custom React Flow node type — without them, edges
// fall back to (0,0) anchor points and visually detach from the node.
export const NODE_W = 180;
export const NODE_H = 64;

export function applyDiffStylingToSide(
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
    // Prefer the human-friendly step type (e.g. "training",
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

export function readSideFromGraph(graph: unknown): SideGraph {
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
export function engineGraphToSide(rawNodes: unknown[]): SideGraph {
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
      // Friendly human label = step type (e.g. "outlier_handling").
      // The raw node_id stays available as `n.id` and is shown as
      // the sub-label / mono badge by the diff card. Using the
      // step_type here also makes the Changes panel readable —
      // otherwise every change row prints a uuid instead of a
      // recognisable step name.
      label: n.step_type ?? n.node_id,
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
export const COL_GAP = 80;
export const ROW_GAP = 36;

export function layoutUnified(
  left: SideGraph,
  right: SideGraph,
  aliases: Map<string, string>,
): { positions: Map<string, { x: number; y: number }>; width: number; height: number } {
  // Canonicalise every id through the alias map so a baseline node
  // and its step-type-matched candidate node end up at one shared
  // position. Without this, runs that persist nodes with fresh
  // uuids would never line up vertically across the two viewers.
  const canon = (id: string) => aliases.get(id) ?? id;
  const allNodeIds = new Set<string>();
  left.nodes.forEach((n) => allNodeIds.add(canon(n.id)));
  right.nodes.forEach((n) => allNodeIds.add(canon(n.id)));

  const inputs = new Map<string, Set<string>>();
  allNodeIds.forEach((id) => inputs.set(id, new Set()));
  [...left.edges, ...right.edges].forEach((e) => {
    const s = canon(e.source);
    const t = canon(e.target);
    if (allNodeIds.has(s) && allNodeIds.has(t)) {
      inputs.get(t)?.add(s);
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

export function applyLayout(
  side: SideGraph,
  positions: Map<string, { x: number; y: number }>,
  aliases: Map<string, string>,
): SideGraph {
  const canon = (id: string) => aliases.get(id) ?? id;
  return {
    nodes: side.nodes.map((n) => ({
      ...n,
      position: positions.get(canon(n.id)) ?? n.position,
    })),
    edges: side.edges,
  };
}
