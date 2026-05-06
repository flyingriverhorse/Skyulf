// Visual diff between two pipeline graphs.
//
// Used by the Experiments → Pipeline Diff tab to highlight what
// changed between two runs (left = baseline, right = candidate).
//
// Diff strategy:
//   - Nodes are matched by `id` (the React Flow id is stable across
//     saves; preserved when loading a job's graph).
//   - For each pair we compare a normalised "config" payload — the
//     node's `data` object minus presentation-only / autosave keys.
//   - Edges are matched by `(source, target, sourceHandle, targetHandle)`.
//
// We never mutate the input graphs; we return a flat report the UI
// then maps onto a duplicated React Flow graph with diff styling.

import type { Node, Edge } from '@xyflow/react';

export type NodeDiffStatus = 'added' | 'removed' | 'modified' | 'unchanged';
export type EdgeDiffStatus = 'added' | 'removed' | 'unchanged';

export interface NodeDiff {
  id: string;
  status: NodeDiffStatus;
  /** Keys that differ between left and right (modified only). */
  changedKeys: string[];
  /** Friendly per-key snippets like `method: zscore → iqr`. Empty when
   *  status is `added`, `removed`, or `unchanged`. */
  changeDescriptions: string[];
  /** Best-effort label sourced from `data.label` / `data.definitionType`. */
  label: string;
}

export interface EdgeDiff {
  id: string;
  status: EdgeDiffStatus;
  source: string;
  target: string;
}

export interface GraphDiff {
  nodes: Map<string, NodeDiff>;
  edges: Map<string, EdgeDiff>;
  /** Baseline-id → candidate-id rename map. Populated when nodes
   *  were paired by `step_type` because their ids differ between
   *  runs (the engine config persists fresh uuids per run). The
   *  diff view uses this so matched nodes share a position in the
   *  unified layout. */
  aliases: Map<string, string>;
  /** Quick counts for the tab badge / summary banner. */
  summary: {
    nodesAdded: number;
    nodesRemoved: number;
    nodesModified: number;
    nodesUnchanged: number;
    /** Pairs that matched by `step_type` rather than id (= the
     *  `aliases` map's size). Surfaces the otherwise-invisible
     *  fallback so users know matched nodes had drifting ids. */
    nodesRenamed: number;
    edgesAdded: number;
    edgesRemoved: number;
    edgesUnchanged: number;
  };
}

// Keys we strip before comparing two `data` blobs. These are
// presentation / runtime artifacts that change between runs even when
// the user-visible config is identical.
const IGNORED_DATA_KEYS = new Set<string>([
  // Per-run execution artifacts written back onto the node by the
  // canvas after a preview run.
  'executionResult',
  'lastRunAt',
  'lastJobId',
  'jobSummaries',
  'bodyPreview',
  // React Flow / store noise that can drift without meaning.
  'selected',
  'dragging',
]);

/** Stable, deterministic JSON for deep-equality comparisons. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableStringify(v)).join(',')}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([k]) => !IGNORED_DATA_KEYS.has(k))
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`;
}

function describeValue(v: unknown): string {
  if (v === null || v === undefined) return '∅';
  if (typeof v === 'string') return v.length > 24 ? `${v.slice(0, 23)}…` : v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return `[${v.length}]`;
  return '{…}';
}

function nodeLabel(node: Node | undefined): string {
  if (!node) return '?';
  const data = (node.data ?? {}) as Record<string, unknown>;
  const label = typeof data.label === 'string' ? data.label : null;
  const def = typeof data.definitionType === 'string' ? data.definitionType : null;
  return label ?? def ?? node.id;
}

function diffNodeData(
  left: Record<string, unknown>,
  right: Record<string, unknown>,
): { changedKeys: string[]; descriptions: string[] } {
  const keys = new Set<string>([...Object.keys(left), ...Object.keys(right)]);
  const changedKeys: string[] = [];
  const descriptions: string[] = [];
  for (const k of keys) {
    if (IGNORED_DATA_KEYS.has(k)) continue;
    const a = left[k];
    const b = right[k];
    if (stableStringify(a) === stableStringify(b)) continue;
    changedKeys.push(k);
    descriptions.push(`${k}: ${describeValue(a)} → ${describeValue(b)}`);
  }
  return { changedKeys, descriptions };
}

function edgeKey(e: Edge): string {
  const sh = e.sourceHandle ?? '';
  const th = e.targetHandle ?? '';
  return `${e.source}|${sh}->${e.target}|${th}`;
}

// Step-type accessor used for fallback matching when ids differ
// between two graphs (e.g. each training run persists its nodes
// with a fresh random `node_id`, so id-only matching would tag
// every node "added"/"removed" even when the pipelines are
// structurally identical).
function stepTypeOf(node: Node | undefined): string | null {
  if (!node) return null;
  const data = (node.data ?? {}) as Record<string, unknown>;
  if (typeof data.definitionType === 'string') return data.definitionType;
  if (typeof data.label === 'string') return data.label;
  return null;
}

export function diffGraphs(
  leftNodes: Node[],
  leftEdges: Edge[],
  rightNodes: Node[],
  rightEdges: Edge[],
): GraphDiff {
  const leftById = new Map(leftNodes.map((n) => [n.id, n]));
  const rightById = new Map(rightNodes.map((n) => [n.id, n]));

  // First pass: pair nodes by id (covers React-Flow snapshots where
  // ids are stable across saves). What's left becomes the candidate
  // pool for the step-type fallback below.
  const matched: Array<{ left: Node; right: Node }> = [];
  const unmatchedLeft: Node[] = [];
  const unmatchedRight: Node[] = [];

  for (const [id, l] of leftById) {
    const r = rightById.get(id);
    if (r) matched.push({ left: l, right: r });
    else unmatchedLeft.push(l);
  }
  for (const [id, r] of rightById) {
    if (!leftById.has(id)) unmatchedRight.push(r);
  }

  // Second pass: pair leftover nodes by `step_type` in declaration
  // order. Each persisted run uses fresh per-node uuids; pairing the
  // Nth "encoding" on the left with the Nth "encoding" on the right
  // lets us still detect that "the encoding step changed param X".
  const leftByType = new Map<string, Node[]>();
  for (const n of unmatchedLeft) {
    const t = stepTypeOf(n);
    if (!t) continue;
    if (!leftByType.has(t)) leftByType.set(t, []);
    leftByType.get(t)!.push(n);
  }
  const stillUnmatchedRight: Node[] = [];
  for (const r of unmatchedRight) {
    const t = stepTypeOf(r);
    const bucket = t ? leftByType.get(t) : undefined;
    const l = bucket?.shift();
    if (l) matched.push({ left: l, right: r });
    else stillUnmatchedRight.push(r);
  }
  // Anything still in `leftByType` is a real removal.
  const stillUnmatchedLeft: Node[] = [];
  for (const bucket of leftByType.values()) stillUnmatchedLeft.push(...bucket);
  // ...plus left nodes that never had a step_type (can't fall back).
  for (const n of unmatchedLeft) {
    if (!stepTypeOf(n) && !matched.some((m) => m.left === n)) {
      stillUnmatchedLeft.push(n);
    }
  }

  const nodes = new Map<string, NodeDiff>();
  let nodesAdded = 0;
  let nodesRemoved = 0;
  let nodesModified = 0;
  let nodesUnchanged = 0;

  // Helper: register the diff entry under both the left and right
  // ids so PipelineDiffView's per-side lookup (`diff.nodes.get(n.id)`)
  // works even when the two sides have different ids for the same
  // structural step.
  const registerPair = (left: Node, right: Node) => {
    const { changedKeys, descriptions } = diffNodeData(
      (left.data ?? {}) as Record<string, unknown>,
      (right.data ?? {}) as Record<string, unknown>,
    );
    const status: NodeDiffStatus = changedKeys.length === 0 ? 'unchanged' : 'modified';
    const entry: NodeDiff = {
      id: right.id,
      status,
      changedKeys,
      changeDescriptions: descriptions,
      label: nodeLabel(right),
    };
    nodes.set(right.id, entry);
    if (left.id !== right.id) nodes.set(left.id, entry);
    if (status === 'unchanged') nodesUnchanged += 1;
    else nodesModified += 1;
  };

  for (const { left, right } of matched) registerPair(left, right);
  for (const l of stillUnmatchedLeft) {
    nodes.set(l.id, {
      id: l.id,
      status: 'removed',
      changedKeys: [],
      changeDescriptions: [],
      label: nodeLabel(l),
    });
    nodesRemoved += 1;
  }
  for (const r of stillUnmatchedRight) {
    nodes.set(r.id, {
      id: r.id,
      status: 'added',
      changedKeys: [],
      changeDescriptions: [],
      label: nodeLabel(r),
    });
    nodesAdded += 1;
  }

  // Build an id-rename map so edge matching can also bridge the
  // baseline ids onto candidate ids. Without this every edge would
  // be flagged added/removed for the same reason node ids drift.
  const leftIdToRightId = new Map<string, string>();
  for (const { left, right } of matched) {
    if (left.id !== right.id) leftIdToRightId.set(left.id, right.id);
  }

  // Translate baseline edge endpoints onto candidate ids so the
  // edge keys line up across runs that use different node uuids.
  const remap = (id: string) => leftIdToRightId.get(id) ?? id;
  const leftEdgesByKey = new Map(
    leftEdges.map((e) => [
      edgeKey({ ...e, source: remap(e.source), target: remap(e.target) } as Edge),
      e,
    ]),
  );
  const rightEdgesByKey = new Map(rightEdges.map((e) => [edgeKey(e), e]));
  const allEdgeKeys = new Set<string>([...leftEdgesByKey.keys(), ...rightEdgesByKey.keys()]);

  const edges = new Map<string, EdgeDiff>();
  let edgesAdded = 0;
  let edgesRemoved = 0;
  let edgesUnchanged = 0;
  for (const key of allEdgeKeys) {
    const l = leftEdgesByKey.get(key);
    const r = rightEdgesByKey.get(key);
    const ref = r ?? l;
    if (!ref) continue;
    if (l && !r) {
      edges.set(ref.id, { id: ref.id, status: 'removed', source: ref.source, target: ref.target });
      edgesRemoved += 1;
    } else if (!l && r) {
      edges.set(ref.id, { id: ref.id, status: 'added', source: ref.source, target: ref.target });
      edgesAdded += 1;
    } else {
      edges.set(ref.id, {
        id: ref.id,
        status: 'unchanged',
        source: ref.source,
        target: ref.target,
      });
      edgesUnchanged += 1;
    }
  }

  return {
    nodes,
    edges,
    aliases: leftIdToRightId,
    summary: {
      nodesAdded,
      nodesRemoved,
      nodesModified,
      nodesUnchanged,
      nodesRenamed: leftIdToRightId.size,
      edgesAdded,
      edgesRemoved,
      edgesUnchanged,
    },
  };
}

/** Tailwind ring class for the given node-diff status. Used by the
 *  diff canvas to outline cards without changing their internals. */
export function nodeDiffRingClass(status: NodeDiffStatus): string {
  switch (status) {
    case 'added':
      return 'ring-2 ring-green-500/70 ring-offset-2 ring-offset-background';
    case 'removed':
      return 'ring-2 ring-red-500/70 ring-offset-2 ring-offset-background opacity-60';
    case 'modified':
      return 'ring-2 ring-amber-500/70 ring-offset-2 ring-offset-background';
    case 'unchanged':
    default:
      return '';
  }
}

/** Stroke colour for diff edges. */
export function edgeDiffStroke(status: EdgeDiffStatus): string | null {
  switch (status) {
    case 'added':
      return '#22c55e';
    case 'removed':
      return '#ef4444';
    case 'unchanged':
    default:
      return null;
  }
}
