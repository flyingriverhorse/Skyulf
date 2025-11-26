import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { ensureArrayOfString } from '../sharedUtils';
import {
  resolveMissingIndicatorSuffix,
  buildMissingIndicatorInsights,
  type MissingIndicatorInsights,
} from '../nodes/missing_indicator/missingIndicatorSettings';
import type { CatalogFlagMap } from './useCatalogFlags';

interface UseMissingIndicatorStateOptions {
  catalogFlags: CatalogFlagMap;
  configState: Record<string, any> | null;
  node: Node;
  availableColumns: string[];
  columnMissingMap: Record<string, number>;
}

export const useMissingIndicatorState = ({
  catalogFlags,
  configState,
  node,
  availableColumns,
  columnMissingMap,
}: UseMissingIndicatorStateOptions) => {
  const { isMissingIndicatorNode } = catalogFlags;
  const activeFlagSuffix = useMemo(() => {
    if (!isMissingIndicatorNode) {
      return '';
    }
    return resolveMissingIndicatorSuffix(configState?.flag_suffix, node?.data?.config?.flag_suffix);
  }, [configState?.flag_suffix, isMissingIndicatorNode, node?.data?.config?.flag_suffix]);

  const missingIndicatorColumns = useMemo(() => {
    if (!isMissingIndicatorNode) {
      return [] as string[];
    }
    return ensureArrayOfString(configState?.columns);
  }, [configState?.columns, isMissingIndicatorNode]);

  const missingIndicatorInsights = useMemo<MissingIndicatorInsights>(() => {
    if (!isMissingIndicatorNode) {
      return { rows: [], flaggedColumnsInDataset: [], conflictCount: 0 };
    }
    return buildMissingIndicatorInsights({
      selectedColumns: missingIndicatorColumns,
      availableColumns,
      columnMissingMap,
      suffix: activeFlagSuffix,
    });
  }, [activeFlagSuffix, availableColumns, columnMissingMap, isMissingIndicatorNode, missingIndicatorColumns]);

  return {
    activeFlagSuffix,
    missingIndicatorColumns,
    missingIndicatorInsights,
  };
};
