import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import { useDatasetSelection } from './useDatasetSelection';
import type { SaveFeedback } from '../types/feedback';

type UseDatasetSelectionHandlerOptions = {
  sourceId?: string | null;
  setIsDirty: Dispatch<SetStateAction<boolean>>;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  isHydratingRef: MutableRefObject<boolean>;
};

export const useDatasetSelectionHandler = ({
  sourceId,
  setIsDirty,
  setSaveFeedback,
  isHydratingRef,
}: UseDatasetSelectionHandlerOptions) => {
  const selectionState = useDatasetSelection(sourceId);
  const { datasets, setActiveSourceId, setSelectedDataset } = selectionState;

  const handleDatasetSelection = useCallback(
    (value: string) => {
      const match = datasets.find((item) => item.source_id === value) ?? null;
      if (!match) {
        setSaveFeedback({ message: 'Dataset unavailable. Choose another option.', tone: 'error' });
        return;
      }

      if (match.is_owned === false) {
        setSaveFeedback({ message: 'Only datasets you own can be selected.', tone: 'error' });
        return;
      }

      setActiveSourceId(value);
      setSelectedDataset(match);
      isHydratingRef.current = true;
      setIsDirty(false);
      setSaveFeedback(null);
    },
    [datasets, isHydratingRef, setActiveSourceId, setIsDirty, setSaveFeedback, setSelectedDataset]
  );

  return {
    ...selectionState,
    handleDatasetSelection,
  };
};
