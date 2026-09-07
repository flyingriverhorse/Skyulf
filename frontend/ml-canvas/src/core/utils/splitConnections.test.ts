import { expect, it } from 'vitest';
import type { Node } from '@xyflow/react';
import { splitOutputHandles, normalizeSplitEdges } from './splitConnections';
import { nextNodePosition } from './nextNodePosition';

/** Build geometry fixtures independently of canvas rendering. */
function node(id: string, x: number, y: number, type = 'imputation_node'): Node {
  return { id, position: { x, y }, width: 240, height: 160, data: { definitionType: type } };
}

it('includes only active split members and always joins X with y', () => {
  // Toggling validation must update the visible bundle without needing new wiring.
  const split = node('split', 0, 0, 'TrainTestSplitter');
  expect(splitOutputHandles(split)).toEqual(['train', 'test']);
  expect(splitOutputHandles({ ...split, data: { ...split.data, validation_size: 0.1 } })).toEqual(['train', 'validation', 'test']);
  expect(splitOutputHandles(node('xy', 0, 0, 'feature_target_split'))).toEqual(['X', 'y']);
});

it('normalizes legacy bundles without joining separate destinations or losing locked state', () => {
  // Saved per-port edges should migrate to one removable unit per downstream input.
  const source = node('split', 0, 0, 'TrainTestSplitter');
  const nodes = [source, node('a', 350, 0), node('b', 350, 240)];
  const edges = [
    { id: 'one', source: 'split', target: 'a', sourceHandle: 'test', targetHandle: 'in' },
    { id: 'two', source: 'split', target: 'a', sourceHandle: 'train', targetHandle: 'in', selected: true, deletable: false },
    { id: 'three', source: 'split', target: 'b', sourceHandle: 'train', targetHandle: 'in' },
  ];
  const normalized = normalizeSplitEdges(nodes, edges);
  expect(normalized).toHaveLength(2);
  expect(normalized[0]).toMatchObject({ id: 'one', sourceHandle: 'train', target: 'a', selected: true, deletable: false });
  expect(normalized[1]?.target).toBe('b');
});

it('places an unobstructed next step close and aligned with its source', () => {
  // Adding a node should not introduce a large gap on an empty canvas.
  const source = node('source', 20, 40);
  const position = nextNodePosition(source, [source]);
  expect(position.y).toBe(source.position.y);
  expect(position.x - (source.position.x + source.width!)).toBeLessThanOrEqual(80);
  expect(position.x).toBeGreaterThan(source.position.x + source.width!);
});

it('searches above crowded nodes instead of drifting far down the canvas', () => {
  // A nearby free slot must win over the previous downward-only scan.
  const source = node('source', 0, 0);
  const blockers = [node('one', 304, 0), node('two', 304, 224), node('three', 304, 448)];
  const position = nextNodePosition(source, [source, ...blockers]);
  expect(position.y).toBeLessThan(0);
  expect(Math.hypot(position.x - 304, position.y)).toBeLessThan(260);
  expect(blockers.every(blocker => position.y + 160 < blocker.position.y)).toBe(true);
});
