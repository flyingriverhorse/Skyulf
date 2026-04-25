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
