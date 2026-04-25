import { describe, it, expect } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { autoLayoutGraph } from './autoLayout';

const mkNode = (id: string): Node => ({
  id,
  type: 'custom',
  position: { x: 0, y: 0 },
  data: {},
});

describe('autoLayoutGraph', () => {
  it('returns the inputs unchanged when there are no nodes', () => {
    const result = autoLayoutGraph([], []);
    expect(result.nodes).toEqual([]);
    expect(result.edges).toEqual([]);
  });

  it('returns edges unchanged (only positions are touched)', () => {
    const nodes = [mkNode('a'), mkNode('b')];
    const edges: Edge[] = [{ id: 'a-b', source: 'a', target: 'b' }];
    const result = autoLayoutGraph(nodes, edges);
    expect(result.edges).toBe(edges);
  });

  it('assigns concrete numeric positions to every node', () => {
    const nodes = [mkNode('a'), mkNode('b'), mkNode('c')];
    const edges: Edge[] = [
      { id: 'a-b', source: 'a', target: 'b' },
      { id: 'b-c', source: 'b', target: 'c' },
    ];
    const result = autoLayoutGraph(nodes, edges);
    for (const n of result.nodes) {
      expect(typeof n.position.x).toBe('number');
      expect(typeof n.position.y).toBe('number');
      expect(Number.isFinite(n.position.x)).toBe(true);
      expect(Number.isFinite(n.position.y)).toBe(true);
    }
  });

  it('lays out nodes left-to-right (downstream node has greater x)', () => {
    const nodes = [mkNode('src'), mkNode('mid'), mkNode('dst')];
    const edges: Edge[] = [
      { id: 'src-mid', source: 'src', target: 'mid' },
      { id: 'mid-dst', source: 'mid', target: 'dst' },
    ];
    const out = autoLayoutGraph(nodes, edges).nodes;
    const byId: Record<string, Node> = {};
    for (const n of out) byId[n.id] = n;
    expect(byId.src!.position.x).toBeLessThan(byId.mid!.position.x);
    expect(byId.mid!.position.x).toBeLessThan(byId.dst!.position.x);
  });

  it('preserves all original node properties (only position changes)', () => {
    const original = mkNode('a');
    original.data = { definitionType: 'imputation_node' };
    const out = autoLayoutGraph([original], []).nodes[0]!;
    expect(out.id).toBe('a');
    expect(out.data).toEqual({ definitionType: 'imputation_node' });
    expect(out.type).toBe('custom');
  });
});
