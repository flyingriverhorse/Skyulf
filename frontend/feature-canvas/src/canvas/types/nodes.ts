import type { SplitTypeKey } from '../constants/splits';

export type FeatureNodeData = {
  label?: string;
  title?: string;
  description?: string;
  isDataset?: boolean;
  isRemovable?: boolean;
  onRemoveNode?: (nodeId: string) => void;
  onOpenSettings?: (nodeId: string) => void;
  inputs?: string[];
  outputs?: string[];
  category?: string;
  parameters?: FeatureNodeParameter[];
  config?: Record<string, any>;
  catalogType?: string;
  backgroundExecutionStatus?: 'idle' | 'loading' | 'success' | 'error';
  activeSplits?: SplitTypeKey[];
  connectedSplits?: SplitTypeKey[];
  hasRequiredConnections?: boolean;
  connectionInfo?: NodeConnectionInfo;
  pendingWarningActive?: boolean;
  pendingWarningReason?: string | null;
  pendingHighlight?: boolean;
};

export type FeatureNodeParameter = {
  name: string;
  label?: string;
  type?: string;
  default?: any;
  [key: string]: any;
};

export type NodeHandleDefinition = {
  key: string;
  label: string;
  position?: number;
  required?: boolean;
  accepts?: ConnectionMatcherKey[];
};

export type ConnectionMatcherKey = 'train' | 'validation' | 'test' | 'model' | 'params';

export type NodeConnectionInfo = {
  inputs?: NodeHandleDefinition[];
  outputs?: NodeHandleDefinition[];
};
