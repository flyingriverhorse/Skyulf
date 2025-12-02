import { useCallback, useRef, useState } from 'react';
import type { PendingConfigurationDetail } from '../../utils/pendingConfiguration';

const buildKey = (details: PendingConfigurationDetail[]): string =>
  details.map((detail) => detail.label.toLowerCase()).join('|');

const sanitizeDetails = (details: PendingConfigurationDetail[]): PendingConfigurationDetail[] => {
  const seen = new Set<string>();
  const result: PendingConfigurationDetail[] = [];
  details.forEach((detail) => {
    const label = detail.label.trim();
    if (!label) {
      return;
    }
    const key = label.toLowerCase();
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    result.push({ label, reason: detail.reason?.trim() || null });
  });
  return result;
};

type PendingToastState = {
  details: PendingConfigurationDetail[];
  issuedAt: number;
};

export const usePendingConfigurationToast = () => {
  const [toastState, setToastState] = useState<PendingToastState | null>(null);
  const dismissedKeyRef = useRef<string | null>(null);

  const showPendingToast = useCallback((details: PendingConfigurationDetail[]) => {
    const sanitized = sanitizeDetails(details);
    if (!sanitized.length) {
      return;
    }
    const payloadKey = buildKey(sanitized);
    if (dismissedKeyRef.current && dismissedKeyRef.current === payloadKey) {
      return;
    }
    setToastState({ details: sanitized, issuedAt: Date.now() });
  }, []);

  const dismissPendingToast = useCallback(() => {
    setToastState((current) => {
      if (current) {
        dismissedKeyRef.current = buildKey(current.details);
      }
      return null;
    });
  }, []);

  const clearPendingToast = useCallback(() => {
    dismissedKeyRef.current = null;
    setToastState(null);
  }, []);

  const pendingToastDetails = toastState?.details ?? [];
  const isPendingToastVisible = Boolean(toastState);

  return {
    pendingToastDetails,
    isPendingToastVisible,
    showPendingToast,
    dismissPendingToast,
    clearPendingToast,
  };
};
