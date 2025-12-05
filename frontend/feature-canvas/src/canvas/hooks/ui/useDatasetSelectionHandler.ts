import { useState, useEffect, useMemo, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchDatasets } from '../../../api';
import type { DatasetSourceSummary } from '../../../api';
import type { SaveFeedback } from '../../types/feedback';

export type DatasetOption = {
  value: string;
  label: string;
  isOwned: boolean;
};

type UseDatasetSelectionHandlerProps = {
  sourceId?: string | null;
  setIsDirty: (isDirty: boolean) => void;
  setSaveFeedback: (feedback: SaveFeedback | null) => void;
  isHydratingRef: React.MutableRefObject<boolean>;
};

export const useDatasetSelectionHandler = ({
  sourceId,
  setIsDirty,
  setSaveFeedback,
  isHydratingRef,
}: UseDatasetSelectionHandlerProps) => {
  const [activeSourceId, setActiveSourceId] = useState<string | null>(sourceId ?? null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetSourceSummary | null>(null);

  const { data: datasets = [], isLoading: isDatasetLoading, error } = useQuery({
    queryKey: ['feature-canvas', 'datasets'],
    queryFn: () => fetchDatasets(100),
    staleTime: 5 * 60 * 1000,
  });

  const datasetErrorMessage = error instanceof Error ? error.message : null;
  const hasDatasets = datasets.length > 0;
  
  const ownedDatasets = useMemo(() => {
    return datasets.filter(d => d.is_owned !== false);
  }, [datasets]);

  const ownedDatasetsCount = ownedDatasets.length;
  const canSelectDatasets = true;

  useEffect(() => {
    if (activeSourceId && datasets.length > 0) {
      const found = datasets.find(d => d.source_id === activeSourceId);
      if (found) {
        setSelectedDataset(found);
      }
    }
  }, [activeSourceId, datasets]);

  const datasetOptions: DatasetOption[] = useMemo(() => {
    return datasets.map(d => ({
      value: d.source_id,
      label: d.name ?? d.source_id,
      isOwned: d.is_owned !== false
    }));
  }, [datasets]);

  const handleDatasetSelection = useCallback((newSourceId: string) => {
    if (newSourceId === activeSourceId) return;
    
    setActiveSourceId(newSourceId);
    setIsDirty(true);
    setSaveFeedback(null);
  }, [activeSourceId, setIsDirty, setSaveFeedback]);

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
    isDatasetLoading,
    datasetErrorMessage,
    handleDatasetSelection,
  };
};
