import { useMutation, type QueryClient } from '@tanstack/react-query';
import { type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import type { FeaturePipelinePayload } from '../../api';
import { savePipeline } from '../../api';
import type { SaveFeedback } from '../types/feedback';

type UsePipelineSaveOptions = {
  queryClient: QueryClient;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  setIsDirty: Dispatch<SetStateAction<boolean>>;
  setActivePipelineId: Dispatch<SetStateAction<number | null>>;
  setActivePipelineName: Dispatch<SetStateAction<string | null>>;
  setActivePipelineUpdatedAt: Dispatch<SetStateAction<string | null>>;
  pendingHistoryIdRef: MutableRefObject<number | null>;
};

type TriggerSaveArgs = {
  sourceId: string;
  payload: FeaturePipelinePayload;
};

type UsePipelineSaveResult = {
  triggerSave: (args: TriggerSaveArgs) => void;
  isSaving: boolean;
};

export const usePipelineSave = ({
  queryClient,
  setSaveFeedback,
  setIsDirty,
  setActivePipelineId,
  setActivePipelineName,
  setActivePipelineUpdatedAt,
  pendingHistoryIdRef,
}: UsePipelineSaveOptions): UsePipelineSaveResult => {
  const { mutate, isPending } = useMutation({
    mutationFn: async ({ sourceId, payload }: TriggerSaveArgs) => savePipeline(sourceId, payload),
    onSuccess: (response, variables) => {
      setSaveFeedback({ message: 'Draft saved successfully.', tone: 'success' });
      setIsDirty(false);
      setActivePipelineId(response.id ?? null);
      setActivePipelineName(response.name ?? null);
      setActivePipelineUpdatedAt(response.updated_at ?? null);
      pendingHistoryIdRef.current = response.id ?? null;
      queryClient.setQueryData(['feature-canvas', 'pipeline', variables.sourceId], response);
      queryClient.invalidateQueries({
        queryKey: ['feature-canvas', 'pipeline-history', variables.sourceId],
        exact: false,
      });
    },
    onError: (error: Error) => {
      setSaveFeedback({ message: error?.message ?? 'Failed to save pipeline.', tone: 'error' });
    },
  });

  return {
    triggerSave: mutate,
    isSaving: isPending,
  };
};