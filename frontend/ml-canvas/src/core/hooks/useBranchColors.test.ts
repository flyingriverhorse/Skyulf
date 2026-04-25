import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { renderHook } from '@testing-library/react';
import { useBranchColors, generateBranchColors } from './useBranchColors';

// Build a tiny linear graph: dataset → preprocessing → terminal.
// Each test stitches its own node/edge fixture so the parallel-vs-merge
// behaviour stays explicit at the call site.
const node = (id: string, definitionType: string, extra: Record<string, unknown> = {}): Node => ({
  id,
  type: 'custom',
  position: { x: 0, y: 0 },
  data: { definitionType, ...extra },
});

const edge = (id: string, source: string, target: string): Edge => ({
  id,
  source,
  target,
});

describe('generateBranchColors', () => {
  it('produces N distinct HSL colors', () => {
    const colors = generateBranchColors(3);
    expect(colors).toHaveLength(3);
    expect(new Set(colors).size).toBe(3);
    expect(colors[0]).toMatch(/^hsl\(\d+, 80%, 65%\)$/);
  });

  it('returns an empty array for count=0', () => {
    expect(generateBranchColors(0)).toEqual([]);
  });
});

describe('useBranchColors', () => {
  it('returns an empty map when no terminals are present', () => {
    const nodes = [node('a', 'imputation_node'), node('b', 'encoding')];
    const edges = [edge('a-b', 'a', 'b')];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.size).toBe(0);
  });

  it('returns an empty map when only one branch exists (nothing to colour)', () => {
    // Single training terminal, single input — no parallel branches.
    const nodes = [
      node('ds', 'dataset_node'),
      node('train', 'basic_training'),
    ];
    const edges = [edge('ds-train', 'ds', 'train')];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.size).toBe(0);
  });

  it('does NOT split a multi-input training terminal in default merge mode', () => {
    // Two upstream sources but execution_mode is left as default ("merge").
    // A single merge branch < 2 → no colouring at all.
    const nodes = [
      node('a', 'dataset_node'),
      node('b', 'dataset_node'),
      node('train', 'basic_training'),
    ];
    const edges = [
      edge('a-train', 'a', 'train'),
      edge('b-train', 'b', 'train'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.size).toBe(0);
  });

  it('splits a multi-input training terminal when execution_mode = "parallel"', () => {
    const nodes = [
      node('a', 'dataset_node'),
      node('b', 'dataset_node'),
      node('train', 'basic_training', { execution_mode: 'parallel', model_type: 'random_forest_classifier' }),
    ];
    const edges = [
      edge('a-train', 'a', 'train'),
      edge('b-train', 'b', 'train'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    // Two branches → both terminal edges get coloured + labelled "Path A/B · Random Forest"
    expect(result.current.size).toBe(2);
    const a = result.current.get('a-train');
    const b = result.current.get('b-train');
    expect(a?.label).toMatch(/^Path A · Random Forest$/);
    expect(b?.label).toMatch(/^Path B · Random Forest$/);
    // Distinct colours per branch
    expect(a?.color).not.toBe(b?.color);
    // Neither edge is shared (each belongs to exactly one branch)
    expect(a?.shared).toBe(false);
    expect(b?.shared).toBe(false);
  });

  it('auto-splits data_preview terminal with 2+ inputs (no toggle needed)', () => {
    const nodes = [
      node('a', 'dataset_node'),
      node('b', 'dataset_node'),
      node('preview', 'data_preview'),
    ];
    const edges = [
      edge('a-preview', 'a', 'preview'),
      edge('b-preview', 'b', 'preview'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.size).toBe(2);
    expect(result.current.get('a-preview')?.label).toMatch(/^Path A/);
    expect(result.current.get('b-preview')?.label).toMatch(/^Path B/);
  });

  it('flags shared upstream edges that feed multiple parallel branches', () => {
    // Common preprocessing node feeds two training terminals → its
    // upstream edge belongs to both branches and should be marked `shared`.
    const nodes = [
      node('ds', 'dataset_node'),
      node('imp', 'imputation_node'),
      node('train_a', 'basic_training', { execution_mode: 'parallel' }),
      node('train_b', 'basic_training', { execution_mode: 'parallel' }),
      node('extra', 'dataset_node'),
    ];
    const edges = [
      edge('ds-imp', 'ds', 'imp'),
      edge('imp-a', 'imp', 'train_a'),
      edge('extra-a', 'extra', 'train_a'),
      edge('imp-b', 'imp', 'train_b'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    // imp itself feeds both terminals: its upstream edge `ds-imp` is shared.
    const dsImp = result.current.get('ds-imp');
    expect(dsImp?.shared).toBe(true);
  });
});
