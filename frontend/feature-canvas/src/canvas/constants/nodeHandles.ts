import type { ConnectionMatcherKey, NodeConnectionInfo, NodeHandleDefinition } from '../types/nodes';
import { getSplitHandlePosition } from './splits';

export const CONNECTION_ACCEPT_MATCHERS: Record<ConnectionMatcherKey, (handleId: string) => boolean> = {
  train: (handleId) => handleId.endsWith('-train'),
  validation: (handleId) => handleId.endsWith('-validation'),
  test: (handleId) => handleId.endsWith('-test'),
  model: (handleId) => handleId.endsWith('-model-out'),
  params: (handleId) => handleId.endsWith('-best-params-out'),
};

export const NODE_HANDLE_CONFIG: Record<string, NodeConnectionInfo> = {
  hyperparameter_tuning: {
    inputs: [
      { key: 'train-in', label: 'Train Split', position: 35, required: true, accepts: ['train'] },
      { key: 'validation-in', label: 'Validation Split', position: 65, required: true, accepts: ['validation'] },
    ],
    outputs: [{ key: 'best-params-out', label: 'Best Params', position: 50 }],
  },
  train_model_draft: {
    inputs: [
      { key: 'train-in', label: 'Train Split', position: 40, required: true, accepts: ['train'] },
      { key: 'params-in', label: 'Best Params', position: 60, required: false, accepts: ['params'] },
    ],
    outputs: [{ key: 'model-out', label: 'Model', position: 50 }],
  },
  model_evaluation: {
    inputs: [
      { key: 'train-in', label: 'Train Split', position: 20, required: false, accepts: ['train'] },
      { key: 'test-in', label: 'Test Split', position: 40, required: false, accepts: ['test'] },
      { key: 'validation-in', label: 'Validation Split', position: 60, required: false, accepts: ['validation'] },
      { key: 'models-in', label: 'Model', position: 80, required: true, accepts: ['model'] },
    ],
  },
  model_registry_overview: {
    inputs: [{ key: 'models-in', label: 'Model', position: 50, required: true, accepts: ['model'] }],
  },
};

export const resolveHandleTopPosition = (
  definition: NodeHandleDefinition,
  index: number,
  total: number
): string => {
  if (typeof definition.position === 'number' && Number.isFinite(definition.position)) {
    const clamped = Math.min(Math.max(definition.position, 5), 95);
    return `${clamped}%`;
  }
  return getSplitHandlePosition(index, total);
};

export const formatHandleDisplayLabel = (definition: NodeHandleDefinition): string => {
  if (definition.required === false) {
    return `${definition.label} (optional)`;
  }
  return `${definition.label} (required)`;
};

export const formatConnectionDescriptor = (definition: NodeHandleDefinition): string =>
  definition.required === false ? `${definition.label} (optional)` : `${definition.label} (required)`;

export const extractHandleKey = (nodeId: string, handleId?: string | null): string | null => {
  if (!handleId || typeof handleId !== 'string') {
    return null;
  }
  const prefix = `${nodeId}-`;
  if (handleId.startsWith(prefix)) {
    return handleId.slice(prefix.length);
  }
  const parts = handleId.split('-');
  if (parts.length >= 2) {
    return parts.slice(-2).join('-');
  }
  return handleId;
};
