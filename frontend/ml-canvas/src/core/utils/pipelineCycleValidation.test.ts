import { describe, it, expect } from 'vitest';
import type { NodeConfigModel } from '../api/client';
import { findCycleIssues } from './pipelineCycleValidation';

const node = (node_id: string, inputs: string[] = []): NodeConfigModel => ({
  node_id,
  step_type: 'Noop',
  params: {},
  inputs,
});

describe('findCycleIssues', () => {
  it('passes a linear chain', () => {
    const nodes = [
      node('load'),
      node('scale', ['load']),
      node('split', ['scale']),
      node('model', ['split']),
    ];
    expect(findCycleIssues(nodes)).toEqual([]);
  });

  it('passes a diamond merge', () => {
    const nodes = [
      node('load'),
      node('left', ['load']),
      node('right', ['load']),
      node('merge', ['left', 'right']),
      node('model', ['merge']),
    ];
    expect(findCycleIssues(nodes)).toEqual([]);
  });

  it('passes disconnected subgraphs', () => {
    const nodes = [node('load_a'), node('model_a', ['load_a']), node('load_b'), node('model_b', ['load_b'])];
    expect(findCycleIssues(nodes)).toEqual([]);
  });

  it('flags a self-loop', () => {
    const issues = findCycleIssues([node('load'), node('model', ['model'])]);
    expect(issues).toHaveLength(1);
    expect(issues[0]!.loopNodeIds).toEqual(['model']);
  });

  it('flags a two-node loop', () => {
    const issues = findCycleIssues([node('a', ['b']), node('b', ['a'])]);
    expect(issues).toHaveLength(1);
    expect(new Set(issues[0]!.loopNodeIds)).toEqual(new Set(['a', 'b']));
  });

  it('names every loop node but not the upstream loader', () => {
    const issues = findCycleIssues([node('load'), node('a', ['c']), node('b', ['a']), node('c', ['b'])]);
    expect(issues).toHaveLength(1);
    expect(new Set(issues[0]!.loopNodeIds)).toEqual(new Set(['a', 'b', 'c']));
  });

  it('excludes downstream innocents from the loop', () => {
    const issues = findCycleIssues([node('a', ['b']), node('b', ['a']), node('downstream', ['b'])]);
    expect(issues).toHaveLength(1);
    expect(new Set(issues[0]!.loopNodeIds)).toEqual(new Set(['a', 'b']));
  });

  it('reports disjoint cycles separately', () => {
    const issues = findCycleIssues([
      node('a', ['b']),
      node('b', ['a']),
      node('x', ['y']),
      node('y', ['x']),
    ]);
    expect(issues).toHaveLength(2);
    const loops = issues.map((issue) => new Set(issue.loopNodeIds));
    expect(loops).toContainEqual(new Set(['a', 'b']));
    expect(loops).toContainEqual(new Set(['x', 'y']));
  });
});
