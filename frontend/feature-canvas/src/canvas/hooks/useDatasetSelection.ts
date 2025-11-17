import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { DatasetSourceSummary } from '../../api';
import { fetchDatasets } from '../../api';

export type DatasetOption = {
  value: string;
  label: string;
  isOwned: boolean;
};

export type UseDatasetSelectionResult = {
  datasets: DatasetSourceSummary[];
  ownedDatasets: DatasetSourceSummary[];
  selectedDataset: DatasetSourceSummary | null;
  setSelectedDataset: React.Dispatch<React.SetStateAction<DatasetSourceSummary | null>>;
  activeSourceId: string | null;
  setActiveSourceId: React.Dispatch<React.SetStateAction<string | null>>;
  datasetOptions: DatasetOption[];
  canSelectDatasets: boolean;
  hasDatasets: boolean;
  ownedDatasetsCount: number;
  isDatasetLoading: boolean;
  datasetErrorMessage: string | null;
};

export const useDatasetSelection = (initialSourceId?: string | null): UseDatasetSelectionResult => {
  const [activeSourceId, setActiveSourceId] = useState<string | null>(initialSourceId ?? null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetSourceSummary | null>(null);

  const datasetsQuery = useQuery({
    queryKey: ['feature-canvas', 'datasets'],
    queryFn: () => fetchDatasets(12),
    staleTime: 5 * 60 * 1000,
  });

  const datasets = datasetsQuery.data ?? [];
  const ownedDatasets = useMemo(
    () => datasets.filter((item) => item?.is_owned !== false),
    [datasets]
  );

  useEffect(() => {
    if (!datasets.length) {
      setSelectedDataset(null);
      return;
    }

    if (activeSourceId) {
      const match = datasets.find((item) => item.source_id === activeSourceId);
      if (match) {
        setSelectedDataset((previous) => (previous?.id === match.id ? previous : match));
        return;
      }
    }

    const fallback = ownedDatasets[0] ?? datasets[0];
    if (!fallback) {
      return;
    }

    if (!activeSourceId) {
      setActiveSourceId(fallback.source_id ?? String(fallback.id));
    }
    setSelectedDataset(fallback);
  }, [activeSourceId, datasets, ownedDatasets]);

  const datasetOptions = useMemo(
    () =>
      datasets.map((item) => ({
        value: item.source_id,
        label: item.name ?? item.source_id,
        isOwned: item.is_owned !== false,
      })),
    [datasets]
  );

  const canSelectDatasets = ownedDatasets.length > 1;
  const hasDatasets = datasets.length > 0;
  const ownedDatasetsCount = ownedDatasets.length;

  const datasetErrorMessage = datasetsQuery.error
    ? (datasetsQuery.error as Error)?.message ?? 'Unable to load datasets'
    : null;

  return {
    datasets,
    ownedDatasets,
    selectedDataset,
    setSelectedDataset,
    activeSourceId,
    setActiveSourceId,
    datasetOptions,
    canSelectDatasets,
    hasDatasets,
    ownedDatasetsCount,
    isDatasetLoading: datasetsQuery.isLoading,
    datasetErrorMessage,
  };
};
