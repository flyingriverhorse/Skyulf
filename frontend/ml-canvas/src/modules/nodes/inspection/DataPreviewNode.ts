import { NodeDefinition } from '../../../core/types/nodes';
import { Table } from 'lucide-react';
import { DataPreviewConfig, DataPreviewComponent, DataPreviewSettings } from './DataPreviewComponents';

export const DataPreviewNode: NodeDefinition<DataPreviewConfig> = {
  type: 'data_preview',
  label: 'Data Preview',
  category: 'Evaluation', // Or Utility
  description: 'Inspect data state and transformations at this point.',
  icon: Table,
  inputs: [{ id: 'in', label: 'Input Data', type: 'any' }],
  outputs: [], // Sink node
  component: DataPreviewComponent,
  settings: DataPreviewSettings,
  validate: () => ({ isValid: true }),
  getDefaultConfig: () => ({}),
};
