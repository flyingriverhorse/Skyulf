import { useCallback, useRef, useState } from 'react';

type ThresholdMutationKind = 'preview' | 'save' | 'enable' | 'disable' | 'clear';

interface ThresholdMutationError {
  kind: ThresholdMutationKind;
  message: string;
}

const thresholdMutationMessage = (kind: ThresholdMutationKind): string => {
  switch (kind) {
    case 'preview':
      return 'Previewing tuned thresholds…';
    case 'save':
      return 'Saving tuned thresholds…';
    case 'enable':
      return 'Enabling tuned thresholds…';
    case 'disable':
      return 'Disabling tuned thresholds…';
    case 'clear':
      return 'Clearing tuned thresholds…';
  }
};

const thresholdMutationLabel = (kind: ThresholdMutationKind): string => {
  switch (kind) {
    case 'preview':
      return 'Preview';
    case 'save':
      return 'Save';
    case 'enable':
      return 'Enable';
    case 'disable':
      return 'Disable';
    case 'clear':
      return 'Clear';
  }
};

/** Preserve mutation progress and retries across view tabs and loading states. */
export function useThresholdMutation() {
  const [pendingThresholdMutation, setPendingThresholdMutation] = useState<ThresholdMutationKind | null>(null);
  const [thresholdMutationError, setThresholdMutationError] = useState<ThresholdMutationError | null>(null);
  const retryThresholdMutationRef = useRef<null | (() => void)>(null);

  const runThresholdMutation = useCallback(async (kind: ThresholdMutationKind, action: () => void | Promise<void>) => {
    setPendingThresholdMutation(kind);
    setThresholdMutationError(null);
    retryThresholdMutationRef.current = () => {
      void runThresholdMutation(kind, action);
    };
    try {
      await Promise.resolve(action());
      retryThresholdMutationRef.current = null;
    } catch (error) {
      setThresholdMutationError({
        kind,
        message: error instanceof Error ? error.message : `Failed to ${thresholdMutationLabel(kind).toLowerCase()} thresholds`,
      });
    } finally {
      setPendingThresholdMutation(current => (current === kind ? null : current));
    }
  }, []);

  const handleRetryThresholdMutation = useCallback(() => {
    if (!retryThresholdMutationRef.current) return;
    retryThresholdMutationRef.current();
  }, []);

  const isThresholdMutationPending = pendingThresholdMutation !== null;
  const pendingThresholdMutationText = pendingThresholdMutation ? thresholdMutationMessage(pendingThresholdMutation) : null;
  const thresholdMutationRetryLabel = thresholdMutationError ? `Retry ${thresholdMutationLabel(thresholdMutationError.kind).toLowerCase()}` : 'Retry';
  return { runThresholdMutation, isThresholdMutationPending, pendingThresholdMutationText, thresholdMutationError, thresholdMutationRetryLabel, handleRetryThresholdMutation };
}

export type ThresholdMutationState = ReturnType<typeof useThresholdMutation>;
