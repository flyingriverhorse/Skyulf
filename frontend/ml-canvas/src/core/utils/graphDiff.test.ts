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
});
