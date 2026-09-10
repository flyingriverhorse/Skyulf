import { Calculator } from 'lucide-react';
import { NodeDefinition } from '../../../core/types/nodes';
import { FeatureGenerationSettings } from './featureGeneration/FeatureGenerationSettings';
import { validateFeatureGeneration } from './featureGeneration/validation';

export const FeatureGenerationNode: NodeDefinition = {
  type: 'FeatureGenerationNode',
  label: 'Feature Generation',
  description: 'Create new features via math, stats, or date extraction.',
  icon: Calculator,
  category: 'Preprocessing',
  inputs: [{ id: 'in', type: 'dataset', label: 'Dataset' }],
  outputs: [{ id: 'out', type: 'dataset', label: 'Enhanced' }],
  validate: validateFeatureGeneration,
  settings: FeatureGenerationSettings,
  bodyPreview: (config) => {
    const n = config.operations?.length ?? 0;
    if (n === 0) return null;
    return `+${n} feature${n === 1 ? '' : 's'}`;
  },
  getDefaultConfig: () => ({
    operations: []
  })
};
