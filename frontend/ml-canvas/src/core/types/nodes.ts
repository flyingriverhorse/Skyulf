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
  message?: string | undefined;
}

export interface NodeSettingsProps<TConfig> {
  config: TConfig;
  onChange: (newConfig: TConfig) => void;
  nodeId?: string;
  /**
   * Whether the Properties panel is maximised. Settings forms can use this
   * to switch between a compact single-column layout (narrow `w-80` panel)
   * and a roomier multi-column layout (full-width expanded panel). Optional —
   * most settings ignore it and render the same in both states.
   */
  isExpanded?: boolean;
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

  // Optional one-line config preview rendered in the node body slot when
  // there is no custom `component` and the backend has not yet emitted a
  // post-run `metadata.summary`. Should return a short string (≤ ~40 chars
  // ideally) describing the user's current configuration, or null if there
  // is nothing meaningful to show yet (so the wrapper falls back to the
  // static `description`). Pure, must not throw.
  bodyPreview?: (config: TConfig) => string | null;

  // Behavior
  validate: (config: TConfig) => ValidationResult;
  getDefaultConfig: () => TConfig;
}
