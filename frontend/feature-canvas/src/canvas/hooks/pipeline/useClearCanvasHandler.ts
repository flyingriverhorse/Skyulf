import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from 'react';
import type { CanvasShellHandle } from '../../types/pipeline';
import type { SaveFeedback } from '../../types/feedback';

type UseClearCanvasHandlerOptions = {
  canvasShellRef: MutableRefObject<CanvasShellHandle | null>;
  canClearCanvas: boolean;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
};

export const useClearCanvasHandler = ({
  canvasShellRef,
  canClearCanvas,
  setSaveFeedback,
}: UseClearCanvasHandlerOptions): (() => void) => {
  return useCallback(() => {
    if (!canClearCanvas) {
      return;
    }

    if (typeof window !== 'undefined') {
      const confirmClear = window.confirm(
        'Clear all nodes and connections from the canvas? This cannot be undone.'
      );
      if (!confirmClear) {
        return;
      }
    }

    canvasShellRef.current?.clearGraph();
    setSaveFeedback((previous) => {
      if (previous?.tone === 'error') {
        return previous;
      }
      return { message: 'Canvas cleared. Unsaved edits pending.', tone: 'info' };
    });
  }, [canvasShellRef, canClearCanvas, setSaveFeedback]);
};
