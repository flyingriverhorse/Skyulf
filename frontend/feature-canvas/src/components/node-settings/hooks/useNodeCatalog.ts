import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchNodeCatalog } from '../../../api';
import type { FeatureNodeCatalogEntry } from '../../../api';

export const useNodeCatalog = () => {
  const { data: nodeCatalog = [], isLoading, error } = useQuery({
    queryKey: ['feature-canvas', 'node-catalog'],
    queryFn: fetchNodeCatalog,
    staleTime: 5 * 60 * 1000,
  });

  const catalogEntryMap = useMemo(() => {
    const map = new Map<string, FeatureNodeCatalogEntry>();
    nodeCatalog.forEach((entry) => {
      if (entry && typeof entry.type === 'string' && entry.type.trim()) {
        map.set(entry.type, entry);
      }
    });
    return map;
  }, [nodeCatalog]);

  return { nodeCatalog, catalogEntryMap, isLoading, error };
};
