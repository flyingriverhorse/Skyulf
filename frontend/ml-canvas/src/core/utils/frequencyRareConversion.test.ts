import { describe, expect, it } from 'vitest';
import { EncodingNode } from '../../modules/nodes/processing/EncodingNode';
import { preprocessingConverters } from './pipelineConversion/preprocessing';

function convert(method: string, overrides = {}) {
  return preprocessingConverters.get('encoding')!({
    id: 'encode', position: { x: 0, y: 0 },
    data: { ...EncodingNode.getDefaultConfig(), definitionType: 'encoding', columns: ['city'], method, ...overrides },
  });
}

describe('category statistics canvas conversion', () => {
  it('sends only frequency controls from shared saved defaults', () => {
    // Hidden options cannot alter the fitted statistics contract.
    expect(convert('frequency', { mode: 'count' })).toEqual({ stepType: 'FrequencyEncoder', params: { columns: ['city'], mode: 'count' } });
  });
  it('sends only rare grouping controls', () => {
    // Group membership must retain explicit thresholds and replacement text.
    expect(convert('rare', { min_count: 3, min_share: 0.25, replacement: 'Grouped' })).toEqual({ stepType: 'RareCategoryGrouping', params: { columns: ['city'], min_count: 3, min_share: 0.25, replacement: 'Grouped' } });
  });
  it('forwards target CV controls without unrelated encoding defaults', () => {
    // The calculator receives user split settings, not one-hot or hash parameters.
    expect(convert('target', { cv: 3, shuffle: false, random_state: null, target_column: 'label' })).toEqual({ stepType: 'TargetEncoder', params: { columns: ['city'], target_column: 'label', smooth: 'auto', target_type: 'auto', cv: 3, shuffle: false, random_state: null } });
  });
  it('retains the historical OneHot fallback for an unknown imported method', () => {
    // Unsupported saved method text must not become an arbitrary core node ID.
    expect(convert('unknown')).toEqual(convert('onehot'));
  });
});
