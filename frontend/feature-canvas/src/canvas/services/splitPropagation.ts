import type { Edge, Node } from 'react-flow-renderer';
import { sanitizeSplitList, SPLIT_TYPE_ORDER, getSplitKeyFromHandle } from '../constants/splits';
import type { SplitTypeKey } from '../constants/splits';
import { extractHandleKey, NODE_HANDLE_CONFIG } from '../constants/nodeHandles';

const SPLIT_PROPAGATION_BLOCKED_TYPES = new Set([
  'train_model_draft',
  'model_registry_overview',
  'model_evaluation',
  'hyperparameter_tuning',
]);

const getNodeRequiredConnections = (catalogType: string): string[] => {
  const config = catalogType ? NODE_HANDLE_CONFIG[catalogType] : undefined;
  if (!config?.inputs?.length) {
    return [];
  }
  return config.inputs
    .filter((definition) => definition.required !== false)
    .map((definition) => definition.key);
};

export const checkNodeConnectionStatus = (nodeId: string, catalogType: string, edges: Edge[]): boolean => {
  const required = getNodeRequiredConnections(catalogType);
  if (required.length === 0) {
    return true;
  }

  const incomingEdges = edges.filter((edge) => edge.target === nodeId);
  const connectedHandles = new Set(
    incomingEdges
      .map((edge) => extractHandleKey(nodeId, edge.targetHandle))
      .filter((value): value is string => Boolean(value))
  );

  return required.every((req) => connectedHandles.has(req));
};

export const computeActiveSplitMap = (nodes: Node[], edges: Edge[]): Map<string, SplitTypeKey[]> => {
  const assignments = new Map<string, Set<SplitTypeKey>>();
  const blockedNodeIds = new Set<string>();

  nodes.forEach((node) => {
    const catalogType = node?.data?.catalogType;
    if (catalogType && SPLIT_PROPAGATION_BLOCKED_TYPES.has(catalogType)) {
      blockedNodeIds.add(node.id);
      return;
    }

    if (catalogType === 'train_test_split') {
      assignments.set(node.id, new Set(SPLIT_TYPE_ORDER));
    }
  });

  let changed = true;
  let iterations = 0;
  const maxIterations = nodes.length * 3;

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    edges.forEach((edge) => {
      if (blockedNodeIds.has(edge.source) || blockedNodeIds.has(edge.target)) {
        return;
      }

      const sourceSplits = assignments.get(edge.source);
      const splitKey = getSplitKeyFromHandle(edge.sourceHandle);

      if (!sourceSplits || sourceSplits.size === 0) {
        return;
      }

      if (!assignments.has(edge.target)) {
        assignments.set(edge.target, new Set());
      }

      const targetSplits = assignments.get(edge.target)!;
      const initialSize = targetSplits.size;

      if (splitKey) {
        if (sourceSplits.has(splitKey)) {
          targetSplits.add(splitKey);
        }
      } else {
        sourceSplits.forEach((split) => targetSplits.add(split));
      }

      if (targetSplits.size > initialSize) {
        changed = true;
      }
    });
  }

  const orderedAssignments = new Map<string, SplitTypeKey[]>();
  assignments.forEach((splitSet, nodeId) => {
    if (!splitSet.size || blockedNodeIds.has(nodeId)) {
      return;
    }

    orderedAssignments.set(
      nodeId,
      SPLIT_TYPE_ORDER.filter((key) => splitSet.has(key))
    );
  });

  return orderedAssignments;
};

export const computeSplitConnectionMap = (edges: Edge[]): Map<string, SplitTypeKey[]> => {
  const connections = new Map<string, Set<SplitTypeKey>>();

  edges.forEach((edge) => {
    const splitKey = getSplitKeyFromHandle(edge.sourceHandle);
    if (!splitKey) {
      return;
    }

    if (!connections.has(edge.source)) {
      connections.set(edge.source, new Set());
    }

    connections.get(edge.source)!.add(splitKey);
  });

  const orderedConnections = new Map<string, SplitTypeKey[]>();
  connections.forEach((splitSet, nodeId) => {
    orderedConnections.set(
      nodeId,
      SPLIT_TYPE_ORDER.filter((key) => splitSet.has(key))
    );
  });

  return orderedConnections;
};
