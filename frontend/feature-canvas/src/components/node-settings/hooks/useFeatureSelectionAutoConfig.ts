import { useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { FeatureSelectionNodeSignal } from '../../../api';
import { type CatalogFlagMap } from './useCatalogFlags';

type UseFeatureSelectionAutoConfigArgs = {
  catalogFlags: CatalogFlagMap;
  featureSelectionSignal: FeatureSelectionNodeSignal | null;
  upstreamTargetColumn: string;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
};

export const useFeatureSelectionAutoConfig = ({
  catalogFlags,
  featureSelectionSignal,
  upstreamTargetColumn,
  setConfigState,
}: UseFeatureSelectionAutoConfigArgs) => {
  const { isFeatureSelectionNode } = catalogFlags;
  useEffect(() => {
    if (!isFeatureSelectionNode) {
      return;
    }

    const normalizedSignalTarget =
      typeof featureSelectionSignal?.target_column === 'string'
        ? featureSelectionSignal.target_column.trim()
        : '';
    const fallbackTarget = typeof upstreamTargetColumn === 'string' ? upstreamTargetColumn.trim() : '';
    const resolvedTarget = normalizedSignalTarget || fallbackTarget;

    if (!resolvedTarget) {
      return;
    }

    setConfigState((previous) => {
      const currentTarget =
        typeof previous?.target_column === 'string' ? previous.target_column.trim() : '';
      if (currentTarget) {
        return previous;
      }
      return {
        ...previous,
        target_column: resolvedTarget,
      };
    });
  }, [featureSelectionSignal?.target_column, isFeatureSelectionNode, setConfigState, upstreamTargetColumn]);

  useEffect(() => {
    if (!isFeatureSelectionNode) {
      return;
    }

    const backendK =
      typeof featureSelectionSignal?.k === 'number' && Number.isFinite(featureSelectionSignal.k)
        ? Math.max(0, Math.trunc(featureSelectionSignal.k))
        : null;
    const selectedCount = Array.isArray(featureSelectionSignal?.selected_columns)
      ? featureSelectionSignal.selected_columns.length
      : null;

    const candidate = (backendK ?? selectedCount) ?? null;
    if (candidate === null || candidate <= 0) {
      return;
    }

    setConfigState((previous) => {
      const rawValue = previous?.k;
      let normalizedCurrent: number | null = null;

      if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
        normalizedCurrent = Math.trunc(rawValue);
      } else if (typeof rawValue === 'string') {
        const parsed = Number(rawValue);
        if (Number.isFinite(parsed)) {
          normalizedCurrent = Math.trunc(parsed);
        }
      }

      if (normalizedCurrent !== null && normalizedCurrent <= candidate) {
        return previous;
      }

      return {
        ...previous,
        k: candidate,
      };
    });
  }, [
    featureSelectionSignal?.k,
    featureSelectionSignal?.selected_columns,
    isFeatureSelectionNode,
    setConfigState,
  ]);
};
