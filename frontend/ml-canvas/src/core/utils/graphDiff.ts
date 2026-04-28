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
  /** Quick counts for the tab badge / summary banner. */
  summary: {
    nodesAdded: number;
    nodesRemoved: number;
    nodesModified: number;
    nodesUnchanged: number;
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

export function diffGraphs(
  leftNodes: Node[],
  leftEdges: Edge[],
  rightNodes: Node[],
  rightEdges: Edge[],
): GraphDiff {
  const leftById = new Map(leftNodes.map((n) => [n.id, n]));
  const rightById = new Map(rightNodes.map((n) => [n.id, n]));
  const allNodeIds = new Set<string>([...leftById.keys(), ...rightById.keys()]);

  const nodes = new Map<string, NodeDiff>();
  let nodesAdded = 0;
  let nodesRemoved = 0;
  let nodesModified = 0;
  let nodesUnchanged = 0;

  for (const id of allNodeIds) {
    const a = leftById.get(id);
    const b = rightById.get(id);
    if (a && !b) {
      nodes.set(id, {
        id,
        status: 'removed',
        changedKeys: [],
        changeDescriptions: [],
        label: nodeLabel(a),
      });
      nodesRemoved += 1;
      continue;
    }
    if (!a && b) {
      nodes.set(id, {
        id,
        status: 'added',
        changedKeys: [],
        changeDescriptions: [],
        label: nodeLabel(b),
      });
      nodesAdded += 1;
      continue;
    }
    if (a && b) {
      const { changedKeys, descriptions } = diffNodeData(
        (a.data ?? {}) as Record<string, unknown>,
        (b.data ?? {}) as Record<string, unknown>,
      );
      if (changedKeys.length === 0) {
        nodes.set(id, {
          id,
          status: 'unchanged',
          changedKeys: [],
          changeDescriptions: [],
          label: nodeLabel(b),
        });
        nodesUnchanged += 1;
      } else {
        nodes.set(id, {
          id,
          status: 'modified',
          changedKeys,
          changeDescriptions: descriptions,
          label: nodeLabel(b),
        });
        nodesModified += 1;
      }
    }
  }

  const leftEdgesByKey = new Map(leftEdges.map((e) => [edgeKey(e), e]));
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
    summary: {
      nodesAdded,
      nodesRemoved,
      nodesModified,
      nodesUnchanged,
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
