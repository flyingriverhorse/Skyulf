import { LucideIcon } from 'lucide-react';
import { NodeDefinition } from '../types/nodes';

// Heterogeneous factory for modeling nodes. Each call site supplies its own
// strongly-typed settings component; we erase the config type at the factory
// boundary because NodeDefinition stores nodes of every shape in one registry.
type AnyConfig = any;

interface CreateNodeConfig {
  type: string;
  label: string;
  description: string;
  icon: LucideIcon;
  settings: React.ComponentType<{ config: AnyConfig; onChange: (next: AnyConfig) => void; nodeId?: string }>;
  defaultConfig?: Record<string, unknown>;
  validate?: (config: AnyConfig) => { isValid: boolean; message?: string };
  bodyPreview?: (config: AnyConfig) => string | null;
  category?: 'Data Source' | 'Preprocessing' | 'Modeling' | 'Evaluation' | 'Utility';
  inputs?: NodeDefinition['inputs'];
  outputs?: NodeDefinition['outputs'];
}

export const createModelingNode = ({
  type,
  label,
  description,
  icon,
  settings,
  defaultConfig = {},
  validate,
  bodyPreview,
  category = 'Modeling',
  inputs = [{ id: 'in', label: 'Training Data', type: 'dataset' }],
  outputs = [{ id: 'model', label: 'Model', type: 'model' }]
}: CreateNodeConfig): NodeDefinition => {
  return {
    type,
    label,
    category,
    description,
    icon,
    inputs,
    outputs,
    settings,
    bodyPreview: bodyPreview || ((config: { model_type?: string; target_column?: string }) => {
      const model = config.model_type;
      const target = config.target_column;
      if (model && target) return `${model} \u2192 ${target}`;
      if (model) return model;
      if (target) return `target: ${target}`;
      return null;
    }),
    validate: validate || ((config: { target_column?: string }) => {
      if (!config.target_column) return { isValid: false, message: 'Target column is required.' };
      return { isValid: true };
    }),
    getDefaultConfig: () => ({
      target_column: '',
      model_type: 'random_forest_classifier',
      ...defaultConfig
    })
  };
};
