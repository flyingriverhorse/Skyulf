/**
 * Single source of truth for the modeling-node `execution_mode` field.
 *
 * Background: training nodes (`basic_training`, `advanced_tuning`) and
 * auto-parallel terminals (`data_preview`) can either merge multiple
 * upstream inputs into one dataset or fan out and run each input as a
 * separate experiment. Before this module the constant set was duplicated
 * across `PropertiesPanel`, `CustomNodeWrapper`, `useBranchColors`, and
 * the per-node settings configs — keep it centralised here so adding a
 * new mode-aware node type is a one-line edit.
 */

import type { Node } from '@xyflow/react';

export type ExecutionMode = 'merge' | 'parallel';

/** The default mode applied when a node has no explicit choice yet. */
export const DEFAULT_EXECUTION_MODE: ExecutionMode = 'merge';

/**
 * Definition types whose users get to pick `execution_mode` explicitly via
 * the Properties Panel toggle. Mirrors `TRAINING_TYPES` in
 * `CustomNodeWrapper.tsx` — kept as a separate Set so the runtime check
 * stays cheap and adding new training-style nodes only touches this file.
 */
export const EXECUTION_MODE_AWARE_TYPES: ReadonlySet<string> = new Set([
  'basic_training',
  'advanced_tuning',
]);

/**
 * Definition types that always run in parallel when wired to ≥2 inputs,
 * regardless of any explicit `execution_mode` value (the user can't pick
 * "merge" for a Data Preview node — there is nothing to merge). Mirrors
 * `AUTO_PARALLEL_STEP_TYPES` in `graph_utils.py` and
 * `AUTO_PARALLEL_TYPES` in `useBranchColors.ts`.
 */
export const AUTO_PARALLEL_TYPES: ReadonlySet<string> = new Set(['data_preview']);

/**
 * Narrow, read-only shape used by helpers below. We deliberately avoid
 * importing the full per-node config types (circular-import risk) and only
 * read the two fields we care about.
 */
export interface ExecutionModeData {
  execution_mode?: ExecutionMode | undefined;
  definitionType?: string | undefined;
}

/** True when this definition type exposes the explicit Merge/Parallel toggle. */
export function supportsExecutionModeToggle(definitionType: string | undefined): boolean {
  return !!definitionType && EXECUTION_MODE_AWARE_TYPES.has(definitionType);
}

/** True when this definition type runs in parallel automatically on multi-input. */
export function isAutoParallelType(definitionType: string | undefined): boolean {
  return !!definitionType && AUTO_PARALLEL_TYPES.has(definitionType);
}

/**
 * Resolve the effective execution mode for a node's data blob. Falls back
 * to {@link DEFAULT_EXECUTION_MODE} when the field is absent or invalid,
 * so callers never have to handle `undefined`.
 */
export function getExecutionMode(data: ExecutionModeData | null | undefined): ExecutionMode {
  const raw = data?.execution_mode;
  return raw === 'parallel' || raw === 'merge' ? raw : DEFAULT_EXECUTION_MODE;
}

/**
 * True when the node should be treated as parallel for runtime / UI
 * purposes — either the user explicitly chose parallel on a mode-aware
 * type, or it's an auto-parallel terminal with multiple incoming sources.
 */
export function isParallelExecution(
  data: ExecutionModeData | null | undefined,
  incomingSourceCount: number,
): boolean {
  const definitionType = data?.definitionType;
  if (isAutoParallelType(definitionType)) return incomingSourceCount > 1;
  if (supportsExecutionModeToggle(definitionType)) {
    return getExecutionMode(data) === 'parallel';
  }
  return false;
}

/**
 * Convenience overload accepting a React Flow node directly. Keeps call
 * sites short where the node object is already in scope (PropertiesPanel,
 * CustomNodeWrapper).
 */
export function getNodeExecutionMode(node: Pick<Node, 'data'>): ExecutionMode {
  return getExecutionMode(node.data as ExecutionModeData);
}
