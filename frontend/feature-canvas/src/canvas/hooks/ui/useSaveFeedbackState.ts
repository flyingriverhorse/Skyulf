import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from 'react';
import type { SaveFeedback } from '../../types/feedback';

type UseSaveFeedbackStateResult = {
  saveFeedback: SaveFeedback | null;
  setSaveFeedback: Dispatch<SetStateAction<SaveFeedback | null>>;
  feedbackIcon: string;
  feedbackClass: string;
};

const resolveFeedbackIcon = (tone?: SaveFeedback['tone']): string => {
  if (tone === 'success') {
    return '✅';
  }
  if (tone === 'error') {
    return '⚠️';
  }
  return '💬';
};

const resolveFeedbackClass = (tone?: SaveFeedback['tone']): string => {
  if (tone === 'error') {
    return 'text-danger';
  }
  if (tone === 'success') {
    return 'text-success';
  }
  return 'text-muted';
};

export const useSaveFeedbackState = (): UseSaveFeedbackStateResult => {
  const [saveFeedback, setSaveFeedback] = useState<SaveFeedback | null>(null);

  useEffect(() => {
    if (!saveFeedback || saveFeedback.tone === 'error' || typeof window === 'undefined') {
      return;
    }
    const timeout = window.setTimeout(
      () => setSaveFeedback(null),
      saveFeedback.tone === 'success' ? 4000 : 2500
    );
    return () => window.clearTimeout(timeout);
  }, [saveFeedback]);

  const feedbackIcon = useMemo(
    () => resolveFeedbackIcon(saveFeedback?.tone),
    [saveFeedback]
  );
  const feedbackClass = useMemo(
    () => resolveFeedbackClass(saveFeedback?.tone),
    [saveFeedback]
  );

  return {
    saveFeedback,
    setSaveFeedback,
    feedbackIcon,
    feedbackClass,
  };
};
