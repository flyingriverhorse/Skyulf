import { useMemo } from 'react';
import type React from 'react';
import type { Node } from 'react-flow-renderer';
import { toHandleKey } from '../utils/connections';

export type ConnectionHandleDescriptor = {
  key: string;
  label: string;
  position?: number;
  required?: boolean;
  accepts?: string[];
};

export type ConnectionInfoSnapshot = {
  inputs?: ConnectionHandleDescriptor[];
  outputs?: ConnectionHandleDescriptor[];
};

type GraphSnapshot = {
  nodes: any[];
  edges: any[];
} | null;

type EvaluationSplitConnectivity = Partial<Record<'train' | 'validation' | 'test', boolean>> | undefined;

type ConnectionReadinessResult = {
  connectionInfo: ConnectionInfoSnapshot | null;
  connectedHandleKeys: Set<string>;
  connectedOutputHandleKeys: Set<string>;
  evaluationSplitConnectivity: EvaluationSplitConnectivity;
  connectionReady: boolean;
  gatedSectionStyle?: React.CSSProperties;
  gatedAriaDisabled?: true;
};

export const useConnectionReadiness = (
  node: Node | undefined,
  graphSnapshot: GraphSnapshot,
): ConnectionReadinessResult => {
  const nodeId = typeof node?.id === 'string' ? node.id : '';
  const connectionInfo = useMemo<ConnectionInfoSnapshot | null>(() => {
    const rawInfo = node?.data?.connectionInfo as ConnectionInfoSnapshot | undefined;
    if (!rawInfo) {
      return null;
    }
    const cloneHandles = (handles?: ConnectionHandleDescriptor[]) =>
      Array.isArray(handles) ? handles.map((handle) => ({ ...handle })) : [];
    return {
      inputs: cloneHandles(rawInfo.inputs),
      outputs: cloneHandles(rawInfo.outputs),
    };
  }, [node]);

  const connectedHandleKeys = useMemo(() => {
    if (!nodeId || !graphSnapshot || !Array.isArray(graphSnapshot.edges)) {
      return new Set<string>();
    }
    const keys = graphSnapshot.edges
      .filter((edge: any) => edge?.target === nodeId)
      .map((edge: any) => toHandleKey(nodeId, edge?.targetHandle))
      .filter((value): value is string => Boolean(value));
    return new Set(keys);
  }, [graphSnapshot, nodeId]);

  const connectedOutputHandleKeys = useMemo(() => {
    if (!nodeId || !graphSnapshot || !Array.isArray(graphSnapshot.edges)) {
      return new Set<string>();
    }
    const keys = graphSnapshot.edges
      .filter((edge: any) => edge?.source === nodeId)
      .map((edge: any) => toHandleKey(nodeId, edge?.sourceHandle))
      .filter((value): value is string => Boolean(value));
    return new Set(keys);
  }, [graphSnapshot, nodeId]);

  const hasRequiredConnections = node?.data?.hasRequiredConnections !== false;

  const requiredInputHandles = useMemo(() => {
    if (!connectionInfo?.inputs?.length) {
      return [] as ConnectionHandleDescriptor[];
    }
    return connectionInfo.inputs.filter((handle) => handle.required !== false);
  }, [connectionInfo?.inputs]);

  const missingRequiredInputs = useMemo(() => {
    if (!requiredInputHandles.length) {
      return [] as ConnectionHandleDescriptor[];
    }
    return requiredInputHandles.filter((handle) => !connectedHandleKeys.has(handle.key));
  }, [connectedHandleKeys, requiredInputHandles]);

  const connectionReady = hasRequiredConnections && missingRequiredInputs.length === 0;

  const gatedSectionStyle = useMemo<React.CSSProperties | undefined>(() => {
    if (connectionReady) {
      return undefined;
    }
    return {
      opacity: 0.45,
      pointerEvents: 'none',
      userSelect: 'none',
    };
  }, [connectionReady]);

  const gatedAriaDisabled = connectionReady ? undefined : true;

  const evaluationSplitConnectivity = useMemo<EvaluationSplitConnectivity>(() => {
    if (!connectionInfo?.inputs || connectionInfo.inputs.length === 0) {
      return undefined;
    }
    let hasRelevant = false;
    const mapping: Partial<Record<'train' | 'validation' | 'test', boolean>> = {};
    connectionInfo.inputs.forEach((handle) => {
      if (!Array.isArray(handle.accepts) || handle.accepts.length === 0) {
        return;
      }
      const isConnected = connectedHandleKeys.has(handle.key);
      handle.accepts.forEach((acceptKey) => {
        if (acceptKey === 'train' || acceptKey === 'validation' || acceptKey === 'test') {
          mapping[acceptKey] = isConnected;
          hasRelevant = true;
        }
      });
    });
    return hasRelevant ? mapping : undefined;
  }, [connectionInfo?.inputs, connectedHandleKeys]);

  return {
    connectionInfo,
    connectedHandleKeys,
    connectedOutputHandleKeys,
    evaluationSplitConnectivity,
    connectionReady,
    gatedSectionStyle,
    gatedAriaDisabled,
  };
};
