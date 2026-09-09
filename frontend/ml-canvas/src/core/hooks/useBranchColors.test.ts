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
  it('orders duplicate-model labels by graph traversal and terminates through cycles', () => {
    /** Canvas path names must follow execution order even when insertion order differs. */
    const nodes = [
      node('later', 'classification', { model_type: 'random_forest_classifier' }),
      node('early', 'classification', { model_type: 'random_forest_classifier' }),
      node('ds', 'dataset_node'), node('prep', 'encoding'), node('cycle', 'scaling'),
    ];
    const edges = [edge('ds-prep', 'ds', 'prep'), edge('ds-early', 'ds', 'early'),
      edge('prep-cycle', 'prep', 'cycle'), edge('cycle-prep', 'cycle', 'prep'),
      edge('cycle-later', 'cycle', 'later')];
    const original = structuredClone({ nodes, edges });
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.get('ds-early')?.label).toBe('Path A · Random Forest #1');
    expect(result.current.get('cycle-later')?.label).toBe('Path B · Random Forest #2');
    expect(result.current.size).toBe(5);
    expect({ nodes, edges }).toEqual(original);
  });

  it('groups split handles and assigns target passthrough to its data branch', () => {
    /** Train/test handles and their target helper must remain one experiment. */
    const nodes = [node('ds', 'dataset_node'), node('fts', 'feature_target_split'),
      node('split', 'train_test_split'), node('other', 'dataset_node'),
      node('train', 'classification', { execution_mode: 'parallel', model_type: 'logistic_regression' })];
    const edges = [edge('ds-fts', 'ds', 'fts'), edge('fts-split', 'fts', 'split'),
      { ...edge('train-handle', 'split', 'train'), sourceHandle: 'train' },
      { ...edge('test-handle', 'split', 'train'), sourceHandle: 'test' },
      { ...edge('target-helper', 'fts', 'train'), sourceHandle: 'y' },
      edge('other-train', 'other', 'train')];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.get('train-handle')?.label).toBe('Path A · Logistic Regression');
    expect(result.current.get('other-train')?.label).toBe('Path B · Logistic Regression');
    expect(result.current.get('test-handle')).toEqual({
      color: result.current.get('train-handle')?.color, label: null, shared: false,
    });
    expect(result.current.get('target-helper')).toEqual(result.current.get('test-handle'));
  });

  it('keeps the first branch color on shared edges and labels only representative terminal edges', () => {
    /** Shared preprocessing must retain a stable color without duplicate path badges. */
    const nodes = [node('ds', 'dataset_node'), node('prep', 'encoding'),
      node('a', 'classification'), node('b', 'regression')];
    const edges = [edge('ds-prep', 'ds', 'prep'), edge('prep-a', 'prep', 'a'),
      edge('prep-a-copy', 'prep', 'a'), edge('prep-b', 'prep', 'b')];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.get('ds-prep')).toEqual({
      color: result.current.get('prep-a')?.color, label: null, shared: true,
    });
    expect(result.current.get('prep-a-copy')?.label).toBeNull();
    expect(result.current.get('prep-b')?.label).toBe('Path B · Regression');
  });

  it('uses source labels for parallel branches and friendly terminal names for preview leaves', () => {
    /** Non-model paths still need meaningful labels aligned with their global tab order. */
    const nodes = [node('a', 'dataset_node', { label: 'Left source' }),
      node('b', 'dataset_node', { title: 'Right source' }),
      node('train', 'classification', { execution_mode: 'parallel' }),
      node('leaf', 'StandardScaler')];
    const edges = [edge('a-train', 'a', 'train'), edge('b-train', 'b', 'train'),
      edge('a-leaf', 'a', 'leaf')];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    expect(result.current.get('a-train')?.label).toBe('Path A · Left source');
    expect(result.current.get('b-train')?.label).toBe('Path B · Right source');
    expect(result.current.get('a-leaf')?.label).toBe('Path C · Standard Scaler');
  });

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
      node('train', 'classification'),
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
      node('train', 'classification'),
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
      node('train', 'classification', { execution_mode: 'parallel', model_type: 'random_forest_classifier' }),
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

  it('does not colour edges feeding a data_preview terminal (preview is its own background job, not a branch terminal)', () => {
    // v0.5.14 deliberately excluded `data_preview` from TERMINAL_TYPES so the
    // canvas no longer paints "Path X · Data Preview" labels on edges feeding
    // a preview node — preview is a standalone evaluation node with its own
    // background job, not a pipeline branch terminal.
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
    expect(result.current.size).toBe(0);
  });

  it('flags shared upstream edges that feed multiple parallel branches', () => {
    // Common preprocessing node feeds two training terminals → its
    // upstream edge belongs to both branches and should be marked `shared`.
    const nodes = [
      node('ds', 'dataset_node'),
      node('imp', 'imputation_node'),
      node('train_a', 'classification', { execution_mode: 'parallel' }),
      node('train_b', 'classification', { execution_mode: 'parallel' }),
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

  it('folds base models that feed an ensemble into the ensemble branch (no separate Path tabs)', () => {
    // Two training nodes act purely as ensemble base learners. They must NOT
    // become their own Path branches — the ensemble is the real terminal and
    // its branch covers the whole `data → model → ensemble` chain. A standalone
    // training node on the same data keeps its own branch so 2 branches remain.
    const nodes = [
      node('ds', 'dataset_node'),
      node('rf', 'classification', { model_type: 'random_forest_classifier' }),
      node('lr', 'classification', { model_type: 'logistic_regression' }),
      node('ens', 'EnsembleNode', { model_type: 'voting_classifier' }),
      node('solo', 'classification', { model_type: 'xgboost_classifier' }),
    ];
    const edges = [
      edge('ds-rf', 'ds', 'rf'),
      edge('ds-lr', 'ds', 'lr'),
      edge('rf-ens', 'rf', 'ens'),
      edge('lr-ens', 'lr', 'ens'),
      edge('ds-solo', 'ds', 'solo'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    // The model→ensemble edges are coloured (part of the ensemble branch) but
    // never carry their own "Path · Random Forest/Logistic Regression" label.
    const rfEns = result.current.get('rf-ens');
    const lrEns = result.current.get('lr-ens');
    expect(rfEns).toBeDefined();
    expect(lrEns).toBeDefined();
    // No base-model branch label leaked onto the spec edges.
    const labels = [...result.current.values()].map(v => v.label).filter(Boolean);
    expect(labels.some(l => /Random Forest|Logistic Regression/.test(l ?? ''))).toBe(false);
    // Exactly two terminals get a Path label: the ensemble (Voting) + the solo
    // XGBoost training node.
    expect(labels.some(l => /· Voting/.test(l ?? ''))).toBe(true);
    expect(labels.some(l => /· Xgboost/.test(l ?? ''))).toBe(true);
  });

  it('keeps the ensemble branch when wired directly to data (no base models)', () => {
    // Ensemble fed straight from a split + a standalone training node → two
    // terminals, so the ensemble earns its own Path tab.
    const nodes = [
      node('ds', 'dataset_node'),
      node('ens', 'EnsembleNode', { model_type: 'stacking_classifier' }),
      node('solo', 'classification', { model_type: 'random_forest_classifier' }),
    ];
    const edges = [
      edge('ds-ens', 'ds', 'ens'),
      edge('ds-solo', 'ds', 'solo'),
    ];
    const { result } = renderHook(() => useBranchColors(nodes, edges));
    const labels = [...result.current.values()].map(v => v.label).filter(Boolean);
    expect(labels.some(l => /· Stacking/.test(l ?? ''))).toBe(true);
    expect(labels.some(l => /· Random Forest/.test(l ?? ''))).toBe(true);
  });
});
