import { useCallback, useEffect, useRef, useState, type MutableRefObject } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchNodeCatalog } from '../../api';
import type { FeatureNodeCatalogEntry } from '../../api';
type UseNodeCatalogDrawerResult = {
  nodeCatalog: FeatureNodeCatalogEntry[];
  catalogEntryMapRef: MutableRefObject<Map<string, FeatureNodeCatalogEntry>>;
  isCatalogOpen: boolean;
  openCatalog: () => void;
  closeCatalog: () => void;
  toggleCatalog: () => void;
  isCatalogLoading: boolean;
  catalogErrorMessage: string | null;
};

export const useNodeCatalogDrawer = (): UseNodeCatalogDrawerResult => {
  const [isCatalogOpen, setIsCatalogOpen] = useState(false);
  const catalogEntryMapRef = useRef<Map<string, FeatureNodeCatalogEntry>>(new Map());

  const nodeCatalogQuery = useQuery({
    queryKey: ['feature-canvas', 'node-catalog'],
    queryFn: fetchNodeCatalog,
    staleTime: 5 * 60 * 1000,
  });

  const nodeCatalog = nodeCatalogQuery.data ?? [];
  const isCatalogLoading = nodeCatalogQuery.isLoading || nodeCatalogQuery.isFetching;
  const catalogErrorMessage = nodeCatalogQuery.error
    ? (nodeCatalogQuery.error as Error)?.message ?? 'Unable to load node catalog'
    : null;

  useEffect(() => {
    const map = new Map<string, FeatureNodeCatalogEntry>();
    nodeCatalog.forEach((entry) => {
      if (entry && typeof entry.type === 'string' && entry.type.trim()) {
        map.set(entry.type, entry);
      }
    });
    catalogEntryMapRef.current = map;
  }, [nodeCatalog]);

  const openCatalog = useCallback(() => setIsCatalogOpen(true), []);
  const closeCatalog = useCallback(() => setIsCatalogOpen(false), []);
  const toggleCatalog = useCallback(() => setIsCatalogOpen((previous) => !previous), []);

  return {
    nodeCatalog,
    catalogEntryMapRef,
    isCatalogOpen,
    openCatalog,
    closeCatalog,
    toggleCatalog,
    isCatalogLoading,
    catalogErrorMessage,
  };
};
