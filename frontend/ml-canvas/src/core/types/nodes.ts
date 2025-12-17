import React from 'react';

export type NodeCategory = 
  | 'Data Source'
  | 'Preprocessing'
  | 'Modeling'
  | 'Evaluation'
  | 'Utility';

export interface PortDefinition {
  id: string;
  label: string;
  type: 'dataset' | 'model' | 'report' | 'any';
}

export interface ValidationResult {
  isValid: boolean;
  message?: string;
}

export interface NodeSettingsProps<TConfig> {
  config: TConfig;
  onChange: (newConfig: TConfig) => void;
  nodeId?: string;
}

export interface NodeDefinition<TConfig = any> {
  // Metadata
  type: string;              // Unique ID, e.g., "imputation_simple"
  label: string;             // Display Name, e.g., "Simple Imputer"
  category: NodeCategory;
  description: string;
  
  // Visual Configuration
  icon?: React.ElementType;      // Lucide Icon / React component
  color?: string;            // Tailwind class or hex
  
  // Logic Configuration
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  
  // Components
  component?: React.JSXElementConstructor<any>; // Custom Node View (optional)
  settings: React.JSXElementConstructor<NodeSettingsProps<TConfig>>; // The Form in the Sidebar
  
  // Behavior
  validate: (config: TConfig) => ValidationResult;
  getDefaultConfig: () => TConfig;
}
