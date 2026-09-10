import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { diffGraphs, uniqueNodeDiffs } from './graphDiff';

const n = (id: string, data: Record<string, unknown> = {}): Node => ({
  id,
  position: { x: 0, y: 0 },
  data,
});

const e = (id: string, source: string, target: string): Edge => ({
  id,
  source,
  target,
});

describe('diffGraphs', () => {
  it('pairs duplicate types in declaration order after stable ids and shares renamed entries', () => {
    // Pair order and alias identity drive both diff canvases and the deduplicated changes list.
    const left = [n('same', { definitionType: 'A' }), n('old1', { definitionType: 'B', z: 1 }), n('old2', { definitionType: 'B', z: 2 }), n('untyped'), n('gone', { definitionType: 'C' })];
    const right = [n('new2', { definitionType: 'B', z: 2 }), n('same', { definitionType: 'A' }), n('new1', { definitionType: 'B', z: 1 })];
    const original = structuredClone({ left, right });
    const diff = diffGraphs(left, [], right, []);
    expect([...diff.aliases]).toEqual([['old1', 'new2'], ['old2', 'new1']]);
    expect([...diff.nodes.keys()]).toEqual(['same', 'new2', 'old1', 'new1', 'old2', 'gone', 'untyped']);
    expect(diff.nodes.get('old1')).toBe(diff.nodes.get('new2'));
    expect(diff.summary).toMatchObject({ nodesModified: 2, nodesRenamed: 2, nodesRemoved: 2 });
    expect({ left, right }).toEqual(original);
  });

  it('keeps last duplicate ids and edge keys while preserving handle distinctions', () => {
    // Map overwrite behavior affects persisted snapshots containing repeated records.
    const left = [n('a', { value: 0 }), n('a', { value: 1 })];
    const edges = [{ ...e('first', 'a', 'b'), sourceHandle: 'x' }, { ...e('last', 'a', 'b'), sourceHandle: 'x' }];
    const diff = diffGraphs(left, edges, [n('a', { value: 1 })], [{ ...e('other', 'a', 'b'), sourceHandle: 'y' }]);
    expect(diff.nodes.get('a')?.status).toBe('unchanged');
    expect([...diff.edges.keys()]).toEqual(['last', 'other']);
    expect(diff.summary).toMatchObject({ edgesRemoved: 1, edgesAdded: 1 });
  });

  it('flags identical graphs as fully unchanged', () => {
    const left = [n('a', { method: 'mean' })];
    const right = [n('a', { method: 'mean' })];
    const diff = diffGraphs(left, [], right, []);
    expect(diff.summary.nodesUnchanged).toBe(1);
    expect(diff.summary.nodesModified).toBe(0);
    expect(diff.summary.nodesAdded).toBe(0);
    expect(diff.summary.nodesRemoved).toBe(0);
  });

  it('detects an added node', () => {
    const diff = diffGraphs([], [], [n('a')], []);
    expect(diff.summary.nodesAdded).toBe(1);
    expect(diff.nodes.get('a')?.status).toBe('added');
  });

  it('detects a removed node', () => {
    const diff = diffGraphs([n('a')], [], [], []);
    expect(diff.summary.nodesRemoved).toBe(1);
    expect(diff.nodes.get('a')?.status).toBe('removed');
  });

  it('detects modified config and lists changed keys', () => {
    const diff = diffGraphs(
      [n('a', { method: 'mean', columns: ['x'] })],
      [],
      [n('a', { method: 'median', columns: ['x'] })],
      [],
    );
    const node = diff.nodes.get('a');
    expect(node?.status).toBe('modified');
    expect(node?.changedKeys).toEqual(['method']);
    expect(node?.changeDescriptions[0]).toContain('method:');
    expect(node?.changeDescriptions[0]).toContain('mean');
    expect(node?.changeDescriptions[0]).toContain('median');
  });

  it('treats NaN and null as different values (F-45)', () => {
    // JSON.stringify renders both NaN and null as "null", so a naive
    // stringify-based equality reports this change as "unchanged".
    const diff = diffGraphs(
      [n('a', { threshold: Number.NaN })],
      [],
      [n('a', { threshold: null })],
      [],
    );
    const node = diff.nodes.get('a');
    expect(node?.status).toBe('modified');
    expect(node?.changedKeys).toEqual(['threshold']);
  });

  it('ignores presentation-only keys (executionResult, lastRunAt, …)', () => {
    const diff = diffGraphs(
      [n('a', { method: 'mean', executionResult: { rows: 100 }, lastRunAt: 'old' })],
      [],
      [n('a', { method: 'mean', executionResult: { rows: 200 }, lastRunAt: 'new' })],
      [],
    );
    expect(diff.nodes.get('a')?.status).toBe('unchanged');
  });

  it('handles edge add/remove via (source,target,handles) key', () => {
    const diff = diffGraphs(
      [n('a'), n('b')],
      [e('e1', 'a', 'b')],
      [n('a'), n('b'), n('c')],
      [e('e1', 'a', 'b'), e('e2', 'b', 'c')],
    );
    expect(diff.summary.edgesAdded).toBe(1);
    expect(diff.summary.edgesRemoved).toBe(0);
    expect(diff.summary.edgesUnchanged).toBe(1);
  });

  it('treats nested-object differences as modifications', () => {
    const diff = diffGraphs(
      [n('a', { params: { lr: 0.01 } })],
      [],
      [n('a', { params: { lr: 0.001 } })],
      [],
    );
    expect(diff.nodes.get('a')?.status).toBe('modified');
  });

  // Regression test for the bug where every node in two structurally
  // identical pipelines was tagged added/removed because each training
  // run persists nodes with fresh per-run uuids. The diff should fall
  // back to step-type matching and surface real config changes only.
  it('matches nodes by step_type when ids drift between runs', () => {
    const left = [
      n('drop_missing-old1', { definitionType: 'drop_missing_columns', threshold: 0.5 }),
      n('encoding-old2', { definitionType: 'encoding', method: 'onehot' }),
    ];
    const right = [
      n('drop_missing-new1', { definitionType: 'drop_missing_columns', threshold: 0.5 }),
      n('encoding-new2', { definitionType: 'encoding', method: 'ordinal' }),
    ];
    const diff = diffGraphs(left, [], right, []);
    expect(diff.summary.nodesAdded).toBe(0);
    expect(diff.summary.nodesRemoved).toBe(0);
    expect(diff.summary.nodesUnchanged).toBe(1);
    expect(diff.summary.nodesModified).toBe(1);
    // Both per-side ids resolve to the same diff entry, so each
    // canvas can colour its own nodes correctly.
    expect(diff.nodes.get('drop_missing-old1')?.status).toBe('unchanged');
    expect(diff.nodes.get('drop_missing-new1')?.status).toBe('unchanged');
    expect(diff.nodes.get('encoding-old2')?.status).toBe('modified');
    expect(diff.nodes.get('encoding-new2')?.status).toBe('modified');
    expect(diff.aliases.get('drop_missing-old1')).toBe('drop_missing-new1');
  });

  it('keeps edges unchanged when their endpoints were renamed by step_type fallback', () => {
    const left = [
      n('a-old', { definitionType: 'load_csv' }),
      n('b-old', { definitionType: 'encoding' }),
    ];
    const right = [
      n('a-new', { definitionType: 'load_csv' }),
      n('b-new', { definitionType: 'encoding' }),
    ];
    const diff = diffGraphs(left, [e('e1', 'a-old', 'b-old')], right, [e('e2', 'a-new', 'b-new')]);
    expect(diff.summary.edgesUnchanged).toBe(1);
    expect(diff.summary.edgesAdded).toBe(0);
    expect(diff.summary.edgesRemoved).toBe(0);
  });

  // F-39: registerPair stores the same NodeDiff object under both the
  // left and right id so each canvas side can look it up. Iterating
  // diff.nodes.values() then yields that object twice, and the diff view
  // double-listed every renamed-and-modified node (plus a React
  // duplicate-key warning). uniqueNodeDiffs must collapse them back to
  // one entry per real change.
  it('uniqueNodeDiffs dedupes a node registered under both renamed ids', () => {
    const left = [
      n('enc-old', { definitionType: 'encoding', method: 'onehot' }),
      n('scale-old', { definitionType: 'scaling', method: 'standard' }),
    ];
    const right = [
      n('enc-new', { definitionType: 'encoding', method: 'ordinal' }),
      n('scale-new', { definitionType: 'scaling', method: 'standard' }),
    ];
    const diff = diffGraphs(left, [], right, []);
    // Both ids resolve for canvas colouring…
    expect(diff.nodes.get('enc-old')).toBe(diff.nodes.get('enc-new'));
    expect(diff.nodes.size).toBe(4);
    // …but the rendered list has exactly one entry per node.
    const unique = uniqueNodeDiffs(diff.nodes);
    expect(unique).toHaveLength(2);
    expect(unique.map(d => d.status).sort()).toEqual(['modified', 'unchanged']);
  });

  // F-44: a real int→string coercion (5 → "5") was detected correctly
  // but rendered as "v: 5 → 5" because describeValue showed numbers and
  // strings identically, so users dismissed a genuine config change.
  // String values must be quoted in descriptions to stay distinguishable.
  it('renders int-to-string coercion as a visibly distinguishable change', () => {
    const diff = diffGraphs(
      [n('a', { v: 5 })],
      [],
      [n('a', { v: '5' })],
      [],
    );
    const node = diff.nodes.get('a');
    expect(node?.status).toBe('modified');
    const desc = node?.changeDescriptions[0];
    expect(desc).toBeDefined();
    // desc is "key: <before> → <after>"; compare only the value halves.
    const valuesPart = desc!.split(': ')[1]!;
    const [before, after] = valuesPart.split(' → ');
    expect(before).not.toBe(after);
  });
});
