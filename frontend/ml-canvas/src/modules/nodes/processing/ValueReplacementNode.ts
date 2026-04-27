import { NodeDefinition } from '../../../core/types/nodes';
import { Replace } from 'lucide-react';
import { ValueReplacementSettings, ValueReplacementConfig } from './ValueReplacementSettings';

export const ValueReplacementNode: NodeDefinition<ValueReplacementConfig> = {
  type: 'value_replacement',
  label: 'Replace Values',
  category: 'Preprocessing',
  description: 'Replace specific values in columns (e.g. -999 to NaN).',
  icon: Replace,
  inputs: [{ id: 'in', label: 'Input Dataset', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Cleaned Dataset', type: 'dataset' }],
  settings: ValueReplacementSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const rules = config.replacements?.length ?? 0;
    if (cols === 0 && rules === 0) return null;
    return `${rules} rule${rules === 1 ? '' : 's'} \u00b7 ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: (config) => {
    if (!config.replacements?.length) return { isValid: true, message: 'No replacements defined (pass-through).' };
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    columns: [],
    replacements: []
  })
};
