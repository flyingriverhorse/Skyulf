import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import type { Node } from 'react-flow-renderer';
import { cloneConfig, stableStringify } from '../../utils/configParsers';

interface UseNodeConfigStateArgs {
  node: Node | null | undefined;
}

interface UseNodeConfigStateResult {
  initialConfig: Record<string, any>;
  configState: Record<string, any>;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
  stableInitialConfig: string;
  stableCurrentConfig: string;
  nodeChangeVersion: number;
  resetToInitialConfig: () => void;
}

export const useNodeConfigState = ({ node }: UseNodeConfigStateArgs): UseNodeConfigStateResult => {
  const nodeId = typeof node?.id === 'string' ? node.id : undefined;

  const initialConfig = useMemo(() => {
    const base = cloneConfig(node?.data?.config ?? {});
    if (!base || typeof base !== 'object' || Array.isArray(base)) {
      return {};
    }
    if (!base.column_overrides || typeof base.column_overrides !== 'object' || Array.isArray(base.column_overrides)) {
      base.column_overrides = {};
    }
    return base;
  }, [node?.id, node?.data?.config]);

  const [configState, setConfigState] = useState<Record<string, any>>(initialConfig);
  const [nodeChangeVersion, setNodeChangeVersion] = useState(0);
  const lastNodeIdRef = useRef<string | undefined>(nodeId);

  useEffect(() => {
    if (lastNodeIdRef.current === nodeId) {
      return;
    }
    lastNodeIdRef.current = nodeId;
    setConfigState(initialConfig);
    setNodeChangeVersion((previous) => previous + 1);
    // Intentionally exclude `initialConfig` from deps to avoid wiping local edits when the
    // upstream node reference changes without the actual node switching.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId]);

  const stableInitialConfig = useMemo(() => stableStringify(initialConfig), [initialConfig]);
  const stableCurrentConfig = useMemo(() => stableStringify(configState), [configState]);

  const resetToInitialConfig = useCallback(() => {
    setConfigState(initialConfig);
  }, [initialConfig]);

  return {
    initialConfig,
    configState,
    setConfigState,
    stableInitialConfig,
    stableCurrentConfig,
    nodeChangeVersion,
    resetToInitialConfig,
  };
};
