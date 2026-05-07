import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { diffGraphs } from './graphDiff';

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
});
