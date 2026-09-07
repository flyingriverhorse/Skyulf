import { beforeAll, describe, expect, it } from 'vitest';
import { initializeRegistry } from '../registry/init';
import { registry } from '../registry/NodeRegistry';
import { groupPreprocessingNodes } from './preprocessingGroups';

beforeAll(() => { initializeRegistry(); });

describe('preprocessing library groups', () => {
  it('assigns every visible preprocessing node exactly once to a named task group', () => {
    // A catalog addition or duplicate mapping must not silently hide or repeat a component.
    const nodes = registry.getAll().filter(node => node.category === 'Preprocessing' && !node.hidden);
    const groups = groupPreprocessingNodes(nodes);
    const types = groups.flatMap(group => group.nodes.map(node => node.type));
    expect(groups.map(group => group.label)).toEqual([
      'Data cleaning', 'Numeric & categorical', 'Feature engineering', 'Text processing', 'Splitting & sampling',
    ]);
    expect(types.sort()).toEqual(nodes.map(node => node.type).sort());
    expect(new Set(types).size).toBe(types.length);
    expect(groups.find(group => group.id === 'split')!.nodes.map(node => node.type)).toEqual(
      expect.arrayContaining(['TrainTestSplitter', 'feature_target_split', 'ResamplingNode']),
    );
  });

  it('keeps future types reachable and omits empty groups and hidden nodes', () => {
    // An unclassified extension still needs a visible route from the library.
    const template = registry.get('imputation_node')!;
    const future = { ...template, type: 'future_preprocessing' };
    const hidden = { ...template, type: 'legacy_preprocessing', hidden: true };
    const groups = groupPreprocessingNodes([future, hidden, registry.get('classification')!]);
    expect(groups).toEqual([{ id: 'other', label: 'Other preprocessing', nodes: [future] }]);
    expect(groupPreprocessingNodes([])).toEqual([]);
  });
});
