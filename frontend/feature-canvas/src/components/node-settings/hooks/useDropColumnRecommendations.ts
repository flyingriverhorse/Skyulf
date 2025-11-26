// Used by NodeSettingsModal when configuring imputer nodes for drop-column guidance.
import { useCallback, useEffect, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { fetchDropColumnRecommendations } from '../../../api';

import { type CatalogFlagMap } from './useCatalogFlags';

type GraphContext = {
  nodes: any[];
  edges: any[];
} | null;

type UseDropColumnRecommendationsArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: GraphContext;
  targetNodeId: string | null;
  setAvailableColumns: Dispatch<SetStateAction<string[]>>;
  setColumnMissingMap: Dispatch<SetStateAction<Record<string, number>>>;
};

type UseDropColumnRecommendationsResult = {
  isFetching: boolean;
  error: string | null;
  refreshDropColumnRecommendations: () => void;
};

export const useDropColumnRecommendations = ({
  catalogFlags,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
  setAvailableColumns,
  setColumnMissingMap,
}: UseDropColumnRecommendationsArgs): UseDropColumnRecommendationsResult => {
  const { isImputerNode } = catalogFlags;
  const shouldLoad = isImputerNode;
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState(0);

  const refreshDropColumnRecommendations = useCallback(() => {
    if (!shouldLoad || !sourceId || !hasReachableSource) {
      return;
    }
    setRequestId((previous) => previous + 1);
  }, [hasReachableSource, shouldLoad, sourceId]);

  useEffect(() => {
    let isActive = true;

    if (!shouldLoad) {
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      setError('Select a dataset to load column metrics.');
      setIsFetching(false);
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      setError('Connect this step to an upstream output to load column metrics.');
      setIsFetching(false);
      return () => {
        isActive = false;
      };
    }

    setIsFetching(true);
    setError(null);

    fetchDropColumnRecommendations(sourceId, {
      graph: graphContext,
      targetNodeId,
    })
      .then((result) => {
        if (!isActive) {
          return;
        }

        const sanitizedMissingMap: Record<string, number> = {};
        if (
          result &&
          typeof result === 'object' &&
          result?.column_missing_map &&
          typeof result.column_missing_map === 'object'
        ) {
          Object.entries(result.column_missing_map).forEach(([rawName, rawValue]) => {
            const name = String(rawName ?? '').trim();
            if (!name) {
              return;
            }
            const numericValue = Number(rawValue);
            if (Number.isNaN(numericValue) || numericValue < 0) {
              return;
            }
            sanitizedMissingMap[name] = Number(numericValue);
          });
        }

        const columnSet = new Set<string>();
        if (Array.isArray(result?.all_columns)) {
          result.all_columns.forEach((column) => {
            const trimmed = String(column ?? '').trim();
            if (trimmed) {
              columnSet.add(trimmed);
            }
          });
        }

        Object.keys(sanitizedMissingMap).forEach((column) => columnSet.add(column));

        const orderedColumns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));
        orderedColumns.forEach((column) => {
          if (!Object.prototype.hasOwnProperty.call(sanitizedMissingMap, column)) {
            sanitizedMissingMap[column] = 0;
          }
        });

        if (orderedColumns.length) {
          setAvailableColumns(orderedColumns);
        }

        if (Object.keys(sanitizedMissingMap).length) {
          setColumnMissingMap((previous) => ({
            ...previous,
            ...sanitizedMissingMap,
          }));
        }
      })
      .catch((fetchError: any) => {
        if (!isActive) {
          return;
        }
        setError(fetchError?.message ?? 'Unable to load column metrics.');
      })
      .finally(() => {
        if (isActive) {
          setIsFetching(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [
    graphContext,
    hasReachableSource,
    requestId,
    setAvailableColumns,
    setColumnMissingMap,
    shouldLoad,
    sourceId,
    targetNodeId,
  ]);

  return {
    isFetching,
    error,
    refreshDropColumnRecommendations,
  };
};
