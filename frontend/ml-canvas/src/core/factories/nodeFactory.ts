import { LucideIcon } from 'lucide-react';
import { NodeDefinition } from '../types/nodes';

// Heterogeneous factory for modeling nodes. `NodeDefinition<TConfig>` is
// already generic; this factory used to erase to `any` internally even
// though nothing forced it to — `TConfig` is threaded through properly so
// TypeScript checks `validate`/`bodyPreview`/`defaultConfig` against the
// same config shape declared by the caller's `settings` component, instead
// of silently accepting anything.
interface CreateNodeConfig<TConfig> {
  type: string;
  label: string;
  description: string;
  icon: LucideIcon;
  settings: React.ComponentType<{ config: TConfig; onChange: (next: TConfig) => void; nodeId?: string }>;
  defaultConfig?: Partial<TConfig>;
  validate?: (config: TConfig) => { isValid: boolean; message?: string };
  bodyPreview?: (config: TConfig) => string | null;
  category?: 'Data Source' | 'Preprocessing' | 'Modeling' | 'Evaluation' | 'Utility';
  inputs?: NodeDefinition['inputs'];
  outputs?: NodeDefinition['outputs'];
}

// Default shape shared by every modeling node created via this factory
// (`BasicTrainingNode`, `AdvancedTuningNode`, `EnsembleNode`) — each caller's
// own `TConfig` still fully applies to their `settings`/`validate`/
// `bodyPreview`, this is just the minimum shape `getDefaultConfig` always
// returns before the caller's `defaultConfig` override is merged in.
interface BaseModelingConfig {
  target_column?: string;
  model_type?: string;
}

export const createModelingNode = <TConfig extends BaseModelingConfig>({
  type,
  label,
  description,
  icon,
  settings,
  defaultConfig,
  validate,
  bodyPreview,
  category = 'Modeling',
  inputs = [{ id: 'in', label: 'Training Data', type: 'dataset' }],
  outputs = [{ id: 'model', label: 'Model', type: 'model' }]
}: CreateNodeConfig<TConfig>): NodeDefinition<TConfig> => {
  return {
    type,
    label,
    category,
    description,
    icon,
    inputs,
    outputs,
    settings,
    bodyPreview: bodyPreview || ((config: TConfig) => {
      const model = config.model_type;
      const target = config.target_column;
      if (model && target) return `${model} \u2192 ${target}`;
      if (model) return model;
      if (target) return `target: ${target}`;
      return null;
    }),
    validate: validate || ((config: TConfig) => {
      if (!config.target_column) return { isValid: false, message: 'Target column is required.' };
      return { isValid: true };
    }),
    getDefaultConfig: () => ({
      target_column: '',
      model_type: 'random_forest_classifier',
      ...defaultConfig
    } as TConfig)
  };
};
