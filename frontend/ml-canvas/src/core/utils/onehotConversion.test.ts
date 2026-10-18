import { describe, expect, it } from 'vitest';
import { EncodingNode } from '../../modules/nodes/processing/EncodingNode';
import { preprocessingConverters } from './pipelineConversion/preprocessing';

describe('one-hot canvas conversion', () => {
  it('submits only fitted-contract options from the real shared encoding defaults', () => {
    // Hidden options from other methods must not invalidate Core saved-state admission.
    const result = preprocessingConverters.get('encoding')!({
      id: 'encode', type: 'custom', position: { x: 0, y: 0 },
      data: { ...EncodingNode.getDefaultConfig(), definitionType: 'encoding', columns: ['city'] },
    });
    expect(result).toEqual({ stepType: 'OneHotEncoder', params: {
      columns: ['city'], drop_first: false, drop_original: true, handle_unknown: 'ignore',
      max_categories: 20, include_missing: false,
    } });
  });

  it('preserves fixed vocabulary order and naming mode without changing other encoders', () => {
    // Categories belong to their selected columns and remain JSON arrays on the wire.
    const data = { method: 'onehot', columns: ['city'], categories: { city: ['Øl', 'Åben'] }, column_names: 'slug', max_categories: null, n_features: 8 };
    const node = { id: 'encode', position: { x: 0, y: 0 }, data };
    expect(preprocessingConverters.get('encoding')!(node).params).toEqual({
      columns: ['city'], categories: { city: ['Øl', 'Åben'] }, column_names: 'slug', max_categories: null,
    });
    expect(preprocessingConverters.get('encoding')!({ ...node, data: { method: 'hash', columns: ['city'], n_features: 8 } }).params.n_features).toBe(8);
  });
});
