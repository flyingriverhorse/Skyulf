import { describe, expect, it } from 'vitest';
import { nodeDisplayNames } from './nodeDisplayNames';

describe('node display names', () => {
  it('disambiguates repeated labels without exposing technical IDs', () => {
    // Screen-reader connection names must distinguish repeated node types.
    const names = nodeDisplayNames(['uuid-a', 'uuid-b'].map(id => ({ id, position: { x: 0, y: 0 }, data: { label: 'Dataset' } })));
    expect([...names.values()]).toEqual(['Dataset (1)', 'Dataset (2)']);
  });

  it('preserves custom names and provides a readable fallback', () => {
    // Imported unknown nodes must not fall back to raw UUIDs in connection labels.
    const names = nodeDisplayNames([
      { id: 'uuid-a', position: { x: 0, y: 0 }, data: { label: 'Customer features' } },
      { id: 'uuid-b', position: { x: 0, y: 0 }, data: {} },
    ]);
    expect([...names.values()]).toEqual(['Customer features', 'Node']);
  });
});
