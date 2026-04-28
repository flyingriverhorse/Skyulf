/**
 * Pipeline Versions API client (L7 — server-side versioning).
 *
 * Mirrors the surface of `core/utils/recentPipelines.ts` so the
 * Toolbar swap is mechanical: same field names (`name`, `pinned`,
 * `datasetName`), same operation order (list → push/create → toggle
 * pin → rename → delete). Versions live server-side, keyed by
 * `dataset_source_id`.
 *
 * The localStorage Recent buffer remains as a read-only fallback for
 * unsaved/offline pipelines; call sites should try the server first
 * and gracefully fall back when the request fails.
 */
import { apiClient } from './client';

import type { Edge, Node } from '@xyflow/react';

/** Server-side pipeline snapshot. Field shape mirrors
 *  `RecentPipelineEntry` so React rendering code is reusable. */
export interface PipelineVersionEntry {
  /** DB primary key. Stable for the lifetime of the row. */
  id: number;
  /** Dataset this snapshot belongs to. */
  datasetId: string;
  /** Monotonically-increasing per-dataset ordinal (v1, v2, …). */
  versionInt: number;
  /** User-facing label (defaults to the pipeline name at save time). */
  name: string;
  /** Optional commit-style note. */
  note?: string;
  /** 'manual' (Save click) or 'auto' (Run hook). */
  kind: 'manual' | 'auto';
  /** Pinned rows float to the top and are not eligible for eviction. */
  pinned: boolean;
  /** Counts captured at snapshot time (cheap to render). */
  nodeCount: number;
  edgeCount: number;
  /** Optional human-readable dataset label. */
  datasetName?: string;
  /** ISO timestamp. */
  createdAt: string;
  /** Full graph payload (engine config or RF snapshot — server is
   *  agnostic). Fetched on demand for restore; kept here so the
   *  Toolbar can populate the canvas without a second round trip. */
  graph: unknown;
}

interface RawVersionRow {
  id: number;
  dataset_source_id: string;
  version_int: number;
  name: string;
  note: string | null;
  kind: string;
  pinned: boolean;
  node_count: number;
  edge_count: number;
  dataset_name: string | null;
  created_at: string;
  graph: unknown;
}

function fromRaw(row: RawVersionRow): PipelineVersionEntry {
  return {
    id: row.id,
    datasetId: row.dataset_source_id,
    versionInt: row.version_int,
    name: row.name,
    ...(row.note ? { note: row.note } : {}),
    kind: row.kind === 'auto' ? 'auto' : 'manual',
    pinned: Boolean(row.pinned),
    nodeCount: row.node_count,
    edgeCount: row.edge_count,
    ...(row.dataset_name ? { datasetName: row.dataset_name } : {}),
    createdAt: row.created_at,
    graph: row.graph,
  };
}

export interface CreateVersionInput {
  name: string;
  graph: unknown;
  note?: string;
  datasetName?: string;
  kind?: 'manual' | 'auto';
  pinned?: boolean;
  /** Convenience: when the caller already has the React Flow nodes/edges
   *  (e.g. straight from the canvas) they can pass them directly and we
   *  pack them into the standard `{nodes, edges}` graph shape. */
  nodes?: Node[];
  edges?: Edge[];
}

export const pipelineVersionsApi = {
  /** List all snapshots for a dataset (pinned first, newest first). */
  async list(datasetId: string): Promise<PipelineVersionEntry[]> {
    const response = await apiClient.get<RawVersionRow[]>(
      `/pipeline/versions/${encodeURIComponent(datasetId)}`,
    );
    return response.data.map(fromRaw);
  },

  /** Create an explicit snapshot. Auto-snapshots from /pipeline/save
   *  happen server-side; use this for "Save as version" UI affordances
   *  or to stamp a snapshot from a successful Run callback. */
  async create(
    datasetId: string,
    input: CreateVersionInput,
  ): Promise<PipelineVersionEntry> {
    const graph =
      input.graph !== undefined
        ? input.graph
        : { nodes: input.nodes ?? [], edges: input.edges ?? [] };
    const body = {
      name: input.name,
      graph,
      ...(input.note ? { note: input.note } : {}),
      ...(input.datasetName ? { dataset_name: input.datasetName } : {}),
      kind: input.kind ?? 'manual',
      pinned: Boolean(input.pinned),
    };
    const response = await apiClient.post<RawVersionRow>(
      `/pipeline/versions/${encodeURIComponent(datasetId)}`,
      body,
    );
    return fromRaw(response.data);
  },

  /** Patch fields. Pass only what changed; server treats missing
   *  fields as "leave alone". */
  async update(
    datasetId: string,
    versionId: number,
    patch: { name?: string; note?: string | null; pinned?: boolean },
  ): Promise<PipelineVersionEntry> {
    const body: Record<string, unknown> = {};
    if (patch.name !== undefined) body.name = patch.name;
    if (patch.note !== undefined) body.note = patch.note ?? '';
    if (patch.pinned !== undefined) body.pinned = patch.pinned;
    const response = await apiClient.patch<RawVersionRow>(
      `/pipeline/versions/${encodeURIComponent(datasetId)}/${versionId}`,
      body,
    );
    return fromRaw(response.data);
  },

  async togglePin(
    datasetId: string,
    versionId: number,
    pinned: boolean,
  ): Promise<PipelineVersionEntry> {
    return this.update(datasetId, versionId, { pinned });
  },

  async rename(
    datasetId: string,
    versionId: number,
    name: string,
  ): Promise<PipelineVersionEntry> {
    return this.update(datasetId, versionId, { name });
  },

  async remove(datasetId: string, versionId: number): Promise<void> {
    await apiClient.delete(
      `/pipeline/versions/${encodeURIComponent(datasetId)}/${versionId}`,
    );
  },
};
