import { expect, it } from 'vitest';
import type { Node } from '@xyflow/react';
import { diffGraphs } from '../../../core/utils/graphDiff';
import { applyDiffStylingToSide, applyLayout, layoutUnified } from './pipelineDiffLayout';

const node = (id: string, data: Record<string, unknown> = {}): Node => ({ id, data, position: { x: 9, y: 8 } });

it('preserves directional status, label fallbacks, and input objects when styling', () => {
  // Both canvases must remain read-only without mutating saved graph snapshots.
  const raw = { nodes: [node('new', { label: 'new', definitionType: 'Scaler' }), node('gone', { label: '' }), node('plain')], edges: [{ id: 'edge', source: 'new', target: 'gone' }] };
  const before = structuredClone(raw);
  const diff = diffGraphs([raw.nodes[1]!], [], [raw.nodes[0]!], raw.edges);
  const left = applyDiffStylingToSide(raw, diff, 'left');
  const right = applyDiffStylingToSide(raw, diff, 'right');
  expect(left.nodes.map(n => n.data)).toEqual([
    { label: 'Scaler', subLabel: 'new', diffStatus: 'unchanged' },
    { label: '', subLabel: 'gone', diffStatus: 'removed' },
    { label: 'plain', subLabel: undefined, diffStatus: 'unchanged' },
  ]);
  expect(right.nodes.map(n => n.data.diffStatus)).toEqual(['added', 'unchanged', 'unchanged']);
  expect(right.nodes[0]).toMatchObject({ type: 'diff', draggable: false, selectable: false });
  expect(left.edges[0]?.style).toEqual({ stroke: '#94a3b8', strokeWidth: 1.5 });
  expect(right.edges[0]?.style).toEqual({ stroke: '#22c55e', strokeWidth: 2 });
  expect(raw).toEqual(before);
});

it('shares renamed coordinates, sorts columns, and retains edge and position identity', () => {
  // Aliased nodes must align across snapshots and layout must not duplicate edges.
  const left = { nodes: [node('z'), node('a'), node('old')], edges: [{ id: 'e', source: 'a', target: 'old' }] };
  const right = { nodes: [node('new')], edges: [] };
  const aliases = new Map([['old', 'new']]);
  const result = layoutUnified(left, right, aliases);
  expect([...result.positions]).toEqual([['a', { x: 0, y: 0 }], ['z', { x: 0, y: 100 }], ['new', { x: 260, y: 50 }]]);
  expect(result).toMatchObject({ width: 520, height: 200 });
  const laidOut = applyLayout(left, result.positions, aliases);
  expect(laidOut.nodes[2]?.position).toBe(result.positions.get('new'));
  expect(laidOut.edges).toBe(left.edges);
  expect(left.nodes[2]?.position).toEqual({ x: 9, y: 8 });
});
