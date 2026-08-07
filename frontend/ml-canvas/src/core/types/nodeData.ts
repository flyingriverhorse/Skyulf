/**
 * Shared shape for `node.data` across every Skyulf canvas node.
 *
 * Every node carries three universal fields plus a node-specific payload.
 * Per-node payloads (e.g. `BasicTrainingConfig`, `OutlierConfig`) live in
 * their own settings modules; this file is intentionally narrow and only
 * captures the shared envelope so call sites can stop reaching for
 * `as { merge_strategy?: string }` style casts.
 *
 * A full per-`definitionType` discriminated union is tracked separately
 * (see `temp/frontend_polish_suggestions_2026.md` #19 / Phase F): doing
 * it properly requires migrating ~25 settings modules to publish a
 * branded config type.
 */

import type { ExecutionMode } from './executionMode';

/**
 * Column-overlap policy the engine accepts. `_get_merge_strategy` in
 * `backend/ml_pipeline/_execution/engine/_merge.py` recognises exactly these
 * two and falls back to `last_wins` for anything else, so the union must not
 * be widened without adding the matching engine branch.
 */
export type MergeStrategy = 'last_wins' | 'first_wins';

/**
 * Universal envelope present on every node's `data`.
 *
 * - `definitionType` — the discriminator key registered in `NodeRegistry`.
 * - `merge_strategy` — column-overlap policy for multi-input merges
 *   (defaulted to `'last_wins'` server-side when absent).
 * - `execution_mode` — modeling-node toggle (`merge` | `parallel`); see
 *   `executionMode.ts` for the canonical helpers.
 */
export interface BaseNodeData {
  definitionType?: string;
  merge_strategy?: MergeStrategy;
  execution_mode?: ExecutionMode;
  // Allow node-specific keys without forcing every consumer to widen.
  [key: string]: unknown;
}

/** Narrow `unknown` / loosely-typed input into `BaseNodeData`. */
export const asBaseNodeData = (data: unknown): BaseNodeData =>
  data && typeof data === 'object' ? (data as BaseNodeData) : {};

/** Read `merge_strategy` with the engine's `'last_wins'` default. */
export const getMergeStrategy = (data: unknown): MergeStrategy =>
  asBaseNodeData(data).merge_strategy ?? 'last_wins';
