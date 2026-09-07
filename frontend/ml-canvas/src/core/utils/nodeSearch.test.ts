import { beforeAll, describe, expect, it } from 'vitest';
import { initializeRegistry } from '../registry/init';
import { registry } from '../registry/NodeRegistry';
import { searchNodes } from './nodeSearch';

beforeAll(() => { initializeRegistry(); });

/** Search the actual catalog so task terms cannot silently point to obsolete node IDs. */
function find(query: string): string[] {
  return searchNodes(registry.getAll(), query).map(node => node.type);
}

describe('node discovery', () => {
  it.each(['normalize', 'normalise', 'standardize numeric', 'min-max', '  NORMALIZE   numeric  '])('finds Scaling for %s without unrelated results', query => {
    // Numeric scaling vocabulary should lead to one relevant component.
    expect(find(query)).toEqual(['scale_numeric_features']);
  });

  it('finds common tasks and narrows multi-word searches', () => {
    // Task words must help newcomers without showing every preprocessing node.
    expect(find('fill blanks')).toEqual(['imputation_node']);
    expect(find('missing values')).toEqual(expect.arrayContaining(['imputation_node', 'MissingIndicator', 'drop_missing_rows']));
    expect(find('missing values')).toHaveLength(3);
    expect(find('predict')).toEqual(expect.arrayContaining(['classification', 'regression', 'text_classification']));
    expect(find('predict numbers')).toEqual(['regression']);
    expect(find('standardize')).toEqual(['scale_numeric_features', 'TextCleaning', 'AliasReplacement']);
    expect(find('predict normalize')).toEqual([]);
  });

  it('preserves name, category, description, and technical type searches', () => {
    // Shared search must retain the existing discovery routes.
    expect(find('Scaling')[0]).toBe('scale_numeric_features');
    expect(find('standard range')).toEqual(['scale_numeric_features']);
    expect(find('scale_numeric_features')).toEqual(['scale_numeric_features']);
    expect(searchNodes(registry.getAll(), 'Modeling').every(node => node.category === 'Modeling')).toBe(true);
  });

  it('ranks a named component above task aliases and excludes hidden definitions', () => {
    // Legacy definitions stay loadable but must never return through search.
    const scaling = registry.get('scale_numeric_features')!;
    const exact = { ...scaling, type: 'named', label: 'Normalize' };
    const hidden = { ...exact, type: 'legacy', hidden: true };
    expect(searchNodes([scaling, hidden, exact], 'normalize')).toEqual([exact, scaling]);
    expect(searchNodes([scaling, hidden, exact], '   ')).toEqual([scaling, exact]);
    expect(searchNodes([hidden], 'normalize')).toEqual([]);
  });
});
