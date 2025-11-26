import { useCallback } from 'react';
import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import type { DatasetSourceSummary } from '../../../api';
import type { PipelineHydrationPayload } from '../../types/pipeline';
import type { SaveFeedback } from '../../types/feedback';
import { formatRelativeTime, formatTimestamp } from '../../utils/time';

type UsePipelineHydrationOptions = {
  applyHydratedSnapshot: (nodes?: any[], edges?: any[]) => { hasCustomNodes: boolean };
  selectedDataset: DatasetSourceSummary | null;
  activeSourceId: string | null;
  setActivePipelineId: Dispatch<SetStateAction<number | null>>;
  setActivePipelineName: Dispatch<SetStateAction<string | null>>;
  setActivePipelineUpdatedAt: Dispatch<SetStateAction<string | null>>;
  setIsDirty: Dispatch<SetStateAction<boolean>>;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  pendingHistoryIdRef: MutableRefObject<number | null>;
  isHydratingRef: MutableRefObject<boolean>;
};

type UsePipelineHydrationResult = {
  handlePipelineHydrated: (payload: PipelineHydrationPayload) => void;
  handlePipelineError: (error: Error) => void;
};

export const usePipelineHydration = ({
  applyHydratedSnapshot,
  selectedDataset,
  activeSourceId,
  setActivePipelineId,
  setActivePipelineName,
  setActivePipelineUpdatedAt,
  setIsDirty,
  setSaveFeedback,
  pendingHistoryIdRef,
  isHydratingRef,
}: UsePipelineHydrationOptions): UsePipelineHydrationResult => {
  const handlePipelineHydrated = useCallback(
    (payload: PipelineHydrationPayload) => {
      const { nodes, edges, pipeline, context } = payload;
      const { hasCustomNodes } = applyHydratedSnapshot(nodes, edges);
      const hydratedPipelineId = pipeline?.id ?? null;

      setActivePipelineId(hydratedPipelineId);
      setActivePipelineName(pipeline?.name ?? null);
      setActivePipelineUpdatedAt(pipeline?.updated_at ?? null);

      if (context === 'stored' && pipeline) {
        if (pendingHistoryIdRef.current === pipeline.id) {
          const relative = formatRelativeTime(pipeline.updated_at);
          const timeLabel = relative ?? formatTimestamp(pipeline.updated_at);
          setSaveFeedback({
            message: `Saved revision “${pipeline.name ?? `#${pipeline.id}`}” (${timeLabel})`,
            tone: 'info',
          });
          pendingHistoryIdRef.current = null;
        } else if (isHydratingRef.current) {
          setSaveFeedback((previous) => {
            if (previous?.tone === 'error') {
              return previous;
            }
            if (hasCustomNodes) {
              return { message: 'Loaded saved pipeline', tone: 'info' };
            }
            return previous;
          });
        }
      } else if (context === 'sample') {
        pendingHistoryIdRef.current = null;
        setSaveFeedback((previous) => {
          if (previous?.tone === 'error') {
            return previous;
          }
          if (!hasCustomNodes) {
            const datasetLabel =
              selectedDataset?.name ?? selectedDataset?.source_id ?? activeSourceId ?? 'demo dataset';
            return {
              message: `Showing starter pipeline for ${datasetLabel}`,
              tone: 'info',
            };
          }
          return previous;
        });
      } else if (context === 'reset') {
        pendingHistoryIdRef.current = null;
      } else if (!pipeline) {
        pendingHistoryIdRef.current = null;
        setSaveFeedback((previous) => (previous?.tone === 'error' ? previous : null));
      }

      if (isHydratingRef.current) {
        setIsDirty(false);
      }

      if (pipeline && pendingHistoryIdRef.current && pendingHistoryIdRef.current !== pipeline.id) {
        pendingHistoryIdRef.current = null;
      }

      isHydratingRef.current = false;
    },
    [
      activeSourceId,
      applyHydratedSnapshot,
      pendingHistoryIdRef,
      selectedDataset,
      setActivePipelineId,
      setActivePipelineName,
      setActivePipelineUpdatedAt,
      setIsDirty,
      setSaveFeedback,
      isHydratingRef,
    ]
  );

  const handlePipelineError = useCallback(
    (error: Error) => {
      setSaveFeedback({
        message: error?.message ?? 'Unable to load saved pipeline. Starting fresh.',
        tone: 'error',
      });
    },
    [setSaveFeedback]
  );

  return { handlePipelineHydrated, handlePipelineError };
};
