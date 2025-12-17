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

export interface NodeDefinition<TConfig = unknown> {
  // Metadata
  type: string;              // Unique ID, e.g., "imputation_simple"
  label: string;             // Display Name, e.g., "Simple Imputer"
  category: NodeCategory;
  description: string;
  
  // Visual Configuration
  icon?: React.FC<unknown>;      // Lucide Icon
  color?: string;            // Tailwind class or hex
  
  // Logic Configuration
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  
  // Components
  component?: React.FC<unknown>; // Custom Node View (optional)
  settings: React.FC<{       // The Form in the Sidebar
    config: TConfig;
    onChange: (newConfig: TConfig) => void;
    nodeId?: string;
  }>;
  
  // Behavior
  validate: (config: TConfig) => ValidationResult;
  getDefaultConfig: () => TConfig;
}
