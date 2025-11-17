import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import type { FeaturePipelineResponse } from '../../api';
import { fetchPipelineHistory } from '../../api';
import type { SaveFeedback } from '../types/feedback';
import { HISTORY_LIMIT } from '../services/layout';

type UsePipelineHistoryOptions = {
  activeSourceId: string | null;
  isDirty: boolean;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  pipelineQueryKey: readonly unknown[];
  pendingHistoryIdRef: MutableRefObject<number | null>;
  isHydratingRef: MutableRefObject<boolean>;
  historyLimit?: number;
};

type UsePipelineHistoryResult = {
  historyItems: FeaturePipelineResponse[];
  isHistoryLoading: boolean;
  historyErrorMessage: string | null;
  refetchHistory: () => Promise<unknown>;
  handleHistorySelection: (pipeline: FeaturePipelineResponse) => void;
};

export const usePipelineHistory = ({
  activeSourceId,
  isDirty,
  setSaveFeedback,
  pipelineQueryKey,
  pendingHistoryIdRef,
  isHydratingRef,
  historyLimit = HISTORY_LIMIT,
}: UsePipelineHistoryOptions): UsePipelineHistoryResult => {
  const queryClient = useQueryClient();

  const historyQuery = useQuery({
    queryKey: ['feature-canvas', 'pipeline-history', activeSourceId],
    queryFn: () => fetchPipelineHistory(activeSourceId as string, historyLimit),
    enabled: Boolean(activeSourceId),
    staleTime: 30 * 1000,
    retry: 1,
  });

  const handleHistorySelection = useCallback(
    (pipeline: FeaturePipelineResponse) => {
      if (!pipeline?.graph) {
        setSaveFeedback({ message: 'Selected revision is missing pipeline data.', tone: 'error' });
        return;
      }

      if (isDirty && typeof window !== 'undefined') {
        const confirmReplace = window.confirm(
          'You have unsaved changes. Replace the canvas with the selected revision?'
        );
        if (!confirmReplace) {
          return;
        }
      }

      pendingHistoryIdRef.current = pipeline.id ?? null;
      isHydratingRef.current = true;
      setSaveFeedback({
        message: `Loading revision “${pipeline.name ?? `#${pipeline.id}`}”…`,
        tone: 'info',
      });
      queryClient.setQueryData(pipelineQueryKey, pipeline);
    },
    [isDirty, isHydratingRef, pendingHistoryIdRef, pipelineQueryKey, queryClient, setSaveFeedback]
  );

  return {
    historyItems: historyQuery.data ?? [],
    isHistoryLoading: historyQuery.isLoading || historyQuery.isFetching,
    historyErrorMessage: historyQuery.error
      ? (historyQuery.error as Error)?.message ?? 'Unable to load history'
      : null,
    refetchHistory: historyQuery.refetch,
    handleHistorySelection,
  };
};