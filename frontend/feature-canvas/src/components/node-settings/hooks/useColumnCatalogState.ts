import { useEffect, useState, useMemo, type Dispatch, type SetStateAction } from 'react';
import type { CatalogFlagMap } from './useCatalogFlags';

type UseColumnCatalogStateArgs = {
  requiresColumnCatalog: boolean;
  catalogFlags: CatalogFlagMap;
  nodeId: string;
  sourceId?: string | null;
  hasReachableSource: boolean;
};

type UseColumnCatalogStateResult = {
  availableColumns: string[];
  setAvailableColumns: Dispatch<SetStateAction<string[]>>;
  columnSearch: string;
  setColumnSearch: Dispatch<SetStateAction<string>>;
  columnMissingMap: Record<string, number>;
  setColumnMissingMap: Dispatch<SetStateAction<Record<string, number>>>;
  columnTypeMap: Record<string, string>;
  setColumnTypeMap: Dispatch<SetStateAction<Record<string, string>>>;
  columnSuggestions: Record<string, string[]>;
  setColumnSuggestions: Dispatch<SetStateAction<Record<string, string[]>>>;
  imputerMissingFilter: number;
  setImputerMissingFilter: Dispatch<SetStateAction<number>>;
  normalizedColumnSearch: string;
  filteredColumnOptions: string[];
};

const resetColumnState = (
  setAvailableColumns: Dispatch<SetStateAction<string[]>>,
  setColumnMissingMap: Dispatch<SetStateAction<Record<string, number>>>,
  setColumnTypeMap: Dispatch<SetStateAction<Record<string, string>>>,
  setColumnSuggestions: Dispatch<SetStateAction<Record<string, string[]>>>,
  setColumnSearch: Dispatch<SetStateAction<string>>,
) => {
  setAvailableColumns([]);
  setColumnMissingMap({});
  setColumnTypeMap({});
  setColumnSuggestions({});
  setColumnSearch('');
};

export const useColumnCatalogState = ({
  requiresColumnCatalog,
  catalogFlags,
  nodeId,
  sourceId,
  hasReachableSource,
}: UseColumnCatalogStateArgs): UseColumnCatalogStateResult => {
  const { isImputerNode } = catalogFlags;
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [columnSearch, setColumnSearch] = useState('');
  const [columnMissingMap, setColumnMissingMap] = useState<Record<string, number>>({});
  const [columnTypeMap, setColumnTypeMap] = useState<Record<string, string>>({});
  const [columnSuggestions, setColumnSuggestions] = useState<Record<string, string[]>>({});
  const [imputerMissingFilter, setImputerMissingFilter] = useState(0);

  useEffect(() => {
    if (!requiresColumnCatalog && !isImputerNode) {
      return;
    }

    if (sourceId && hasReachableSource) {
      return;
    }

    resetColumnState(
      setAvailableColumns,
      setColumnMissingMap,
      setColumnTypeMap,
      setColumnSuggestions,
      setColumnSearch,
    );
  }, [hasReachableSource, isImputerNode, requiresColumnCatalog, sourceId]);

  useEffect(() => {
    setImputerMissingFilter(0);
  }, [isImputerNode, nodeId]);

  const normalizedColumnSearch = useMemo(() => columnSearch.trim().toLowerCase(), [columnSearch]);

  const filteredColumnOptions = useMemo(() => {
    if (!normalizedColumnSearch) {
      return availableColumns;
    }
    return availableColumns.filter((column) => column.toLowerCase().includes(normalizedColumnSearch));
  }, [availableColumns, normalizedColumnSearch]);

  return {
    availableColumns,
    setAvailableColumns,
    columnSearch,
    setColumnSearch,
    columnMissingMap,
    setColumnMissingMap,
    columnTypeMap,
    setColumnTypeMap,
    columnSuggestions,
    setColumnSuggestions,
    imputerMissingFilter,
    setImputerMissingFilter,
    normalizedColumnSearch,
    filteredColumnOptions,
  };
};
