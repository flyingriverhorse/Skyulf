import { useState, useEffect, useMemo } from 'react';
import type { FeatureMathNodeSignal } from '../../../api';
import {
  buildFeatureMathSummaries,
  FeatureMathOperationDraft,
  normalizeFeatureMathOperations,
  FeatureMathOperationSummary,
} from '../nodes/feature_math/featureMathSettings';

export const useFeatureMathState = (
  isFeatureMathNode: boolean,
  configState: Record<string, any> | null,
  featureMathSignals: FeatureMathNodeSignal[]
) => {
  const [collapsedFeatureMath, setCollapsedFeatureMath] = useState<Set<string>>(() => new Set());

  const featureMathOperations = useMemo<FeatureMathOperationDraft[]>(() => {
    if (!isFeatureMathNode) {
      return [];
    }
    return normalizeFeatureMathOperations(configState?.operations ?? []);
  }, [configState?.operations, isFeatureMathNode]);

  const featureMathSummaries = useMemo<FeatureMathOperationSummary[]>(
    () => (isFeatureMathNode ? buildFeatureMathSummaries(featureMathOperations, featureMathSignals) : []),
    [featureMathOperations, featureMathSignals, isFeatureMathNode],
  );

  useEffect(() => {
    if (!isFeatureMathNode) {
      setCollapsedFeatureMath(() => new Set());
      return;
    }
    setCollapsedFeatureMath((previous) => {
      if (!previous.size) {
        return previous;
      }
      const validIds = new Set(featureMathOperations.map((operation) => operation.id));
      const next = new Set<string>();
      previous.forEach((id) => {
        if (validIds.has(id)) {
          next.add(id);
        }
      });
      return next.size === previous.size ? previous : next;
    });
  }, [featureMathOperations, isFeatureMathNode]);

  return {
    featureMathOperations,
    featureMathSummaries,
    collapsedFeatureMath,
    setCollapsedFeatureMath,
  };
};
