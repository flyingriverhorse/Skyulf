import { useMemo, useCallback } from 'react';
import { DEFAULT_KEEP_STRATEGY, type KeepStrategy } from '../../nodes/remove_duplicates/removeDuplicatesSettings';
import { useCatalogFlags } from '../core/useCatalogFlags';
import type { Node } from 'react-flow-renderer';

type UseRemoveDuplicatesStateArgs = {
  node: Node;
  nodeId: string;
  configState: Record<string, any>;
  setConfigState: any;
};

export const useRemoveDuplicatesState = ({
  node,
  nodeId,
  configState,
  setConfigState,
}: UseRemoveDuplicatesStateArgs) => {
  const { isRemoveDuplicatesNode } = useCatalogFlags(node);
  const removeDuplicatesKeepSelectId = `${nodeId || 'node'}-remove-duplicates-keep`;

  const removeDuplicatesKeep = useMemo<KeepStrategy>(() => {
    if (!isRemoveDuplicatesNode) {
      return DEFAULT_KEEP_STRATEGY;
    }
    const raw = typeof configState?.keep === 'string' ? configState.keep.trim().toLowerCase() : '';
    if (raw === 'last' || raw === 'none') {
      return raw;
    }
    return DEFAULT_KEEP_STRATEGY;
  }, [configState?.keep, isRemoveDuplicatesNode]);

  const handleRemoveDuplicatesKeepChange = useCallback(
    (value: KeepStrategy) => {
      setConfigState((previous: any) => ({
        ...previous,
        keep: value,
      }));
    },
    [setConfigState],
  );

  return {
    removeDuplicatesKeepSelectId,
    removeDuplicatesKeep,
    handleRemoveDuplicatesKeepChange,
  };
};
