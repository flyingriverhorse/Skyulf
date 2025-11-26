import { useEffect, useCallback } from 'react';
import { useLabelEncodingRecommendations } from './useLabelEncodingRecommendations';
import { useTargetEncodingRecommendations } from './useTargetEncodingRecommendations';
import { useHashEncodingRecommendations } from './useHashEncodingRecommendations';
import { useOrdinalEncodingRecommendations } from './useOrdinalEncodingRecommendations';
import { useDummyEncodingRecommendations } from './useDummyEncodingRecommendations';
import { useOneHotEncodingRecommendations } from './useOneHotEncodingRecommendations';
import { useTargetEncodingDefaults } from './useTargetEncodingDefaults';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

type UseEncodingRecommendationsStateProps = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: any;
  node: any;
  configState: any;
  setConfigState: (updater: any) => void;
  nodeChangeVersion: number;
};

export const useEncodingRecommendationsState = ({
  catalogFlags,
  sourceId,
  hasReachableSource,
  graphContext,
  node,
  configState,
  setConfigState,
  nodeChangeVersion,
}: UseEncodingRecommendationsStateProps) => {
  const {
    isLabelEncodingNode,
    isTargetEncodingNode,
    isHashEncodingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
  } = catalogFlags;

  const targetNodeId = node?.id ?? null;

  const {
    isFetching: isFetchingLabelEncoding,
    error: labelEncodingError,
    suggestions: labelEncodingSuggestions,
    metadata: labelEncodingMetadata,
  } = useLabelEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  const {
    isFetching: isFetchingTargetEncoding,
    error: targetEncodingError,
    suggestions: targetEncodingSuggestions,
    metadata: targetEncodingMetadata,
  } = useTargetEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  useTargetEncodingDefaults({
    catalogFlags,
    enableGlobalFallbackDefault: targetEncodingMetadata.enableGlobalFallbackDefault,
    encodeMissing: configState?.encode_missing,
    handleUnknown: configState?.handle_unknown,
    setConfigState,
    nodeChangeVersion,
  });

  const {
    isFetching: isFetchingHashEncoding,
    error: hashEncodingError,
    suggestions: hashEncodingSuggestions,
    metadata: hashEncodingMetadata,
  } = useHashEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  const {
    isFetching: isFetchingOrdinalEncoding,
    error: ordinalEncodingError,
    suggestions: ordinalEncodingSuggestions,
    metadata: ordinalEncodingMetadata,
  } = useOrdinalEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  const {
    isFetching: isFetchingDummyEncoding,
    error: dummyEncodingError,
    suggestions: dummyEncodingSuggestions,
    metadata: dummyEncodingMetadata,
  } = useDummyEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  const {
    isFetching: isFetchingOneHotEncoding,
    error: oneHotEncodingError,
    suggestions: oneHotEncodingSuggestions,
    metadata: oneHotEncodingMetadata,
  } = useOneHotEncodingRecommendations({
    catalogFlags,
    sourceId,
    hasReachableSource,
    graphContext,
    targetNodeId,
  });

  // Auto-detect effects
  useEffect(() => {
    if (isLabelEncodingNode && configState?.auto_detect && labelEncodingMetadata.autoDetectDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        columns: labelEncodingMetadata.autoDetectDefault,
      }));
    }
  }, [configState?.auto_detect, isLabelEncodingNode, labelEncodingMetadata.autoDetectDefault, setConfigState]);

  useEffect(() => {
    if (isTargetEncodingNode && configState?.auto_detect && targetEncodingMetadata.autoDetectDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        columns: targetEncodingMetadata.autoDetectDefault,
      }));
    }
  }, [configState?.auto_detect, isTargetEncodingNode, setConfigState, targetEncodingMetadata.autoDetectDefault]);

  useEffect(() => {
    if (isHashEncodingNode && configState?.auto_detect && hashEncodingMetadata.autoDetectDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        columns: hashEncodingMetadata.autoDetectDefault,
      }));
    }
  }, [configState?.auto_detect, hashEncodingMetadata.autoDetectDefault, isHashEncodingNode, setConfigState]);

  useEffect(() => {
    if (isHashEncodingNode && hashEncodingMetadata.suggestedBucketDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        n_buckets: hashEncodingMetadata.suggestedBucketDefault,
      }));
    }
  }, [hashEncodingMetadata.suggestedBucketDefault, isHashEncodingNode, setConfigState]);

  useEffect(() => {
    if (isTargetEncodingNode && targetEncodingMetadata.enableGlobalFallbackDefault) {
      setConfigState((prev: any) => {
        const updates: any = {};
        if (prev.encode_missing === undefined) updates.encode_missing = true;
        if (prev.handle_unknown === undefined) updates.handle_unknown = 'value';
        return Object.keys(updates).length > 0 ? { ...prev, ...updates } : prev;
      });
    }
  }, [
    configState?.encode_missing,
    configState?.handle_unknown,
    isTargetEncodingNode,
    setConfigState,
    targetEncodingMetadata.enableGlobalFallbackDefault,
  ]);

  useEffect(() => {
    if (isOrdinalEncodingNode && configState?.auto_detect && ordinalEncodingMetadata.autoDetectDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        columns: ordinalEncodingMetadata.autoDetectDefault,
      }));
    }
  }, [configState?.auto_detect, isOrdinalEncodingNode, ordinalEncodingMetadata.autoDetectDefault, setConfigState]);

  useEffect(() => {
    if (isOrdinalEncodingNode && ordinalEncodingMetadata.enableUnknownDefault) {
      setConfigState((prev: any) => {
        if (prev.handle_unknown === undefined) {
          return { ...prev, handle_unknown: 'use_encoded_value', unknown_value: -1 };
        }
        return prev;
      });
    }
  }, [configState?.handle_unknown, isOrdinalEncodingNode, ordinalEncodingMetadata.enableUnknownDefault, setConfigState]);

  useEffect(() => {
    if (isDummyEncodingNode && configState?.auto_detect && dummyEncodingMetadata.autoDetectDefault) {
      setConfigState((prev: any) => ({
        ...prev,
        columns: dummyEncodingMetadata.autoDetectDefault,
      }));
    }
  }, [configState?.auto_detect, dummyEncodingMetadata.autoDetectDefault, isDummyEncodingNode, setConfigState]);

  // Handlers
  const handleApplyLabelEncodingRecommended = useCallback((columns: string[]) => {
    setConfigState((prev: any) => ({ ...prev, columns }));
  }, [setConfigState]);

  const handleApplyTargetEncodingRecommended = useCallback(
    (columns: string[]) => {
      setConfigState((prev: any) => ({ ...prev, columns }));
    },
    [setConfigState]
  );

  const handleApplyHashEncodingRecommended = useCallback(
    (columns: string[]) => {
      setConfigState((prev: any) => ({ ...prev, columns }));
    },
    [setConfigState]
  );

  const handleApplyOrdinalEncodingRecommended = useCallback(
    (columns: string[]) => {
      setConfigState((prev: any) => ({ ...prev, columns }));
    },
    [setConfigState]
  );

  const handleApplyDummyEncodingRecommended = useCallback(
    (columns: string[]) => {
      setConfigState((prev: any) => ({ ...prev, columns }));
    },
    [setConfigState]
  );

  const handleApplyOneHotEncodingRecommended = useCallback(
    (columns: string[]) => {
      setConfigState((prev: any) => ({ ...prev, columns }));
    },
    [setConfigState]
  );

  return {
    labelEncoding: {
      isFetching: isFetchingLabelEncoding,
      error: labelEncodingError,
      suggestions: labelEncodingSuggestions,
      metadata: labelEncodingMetadata,
      handleApplyRecommended: handleApplyLabelEncodingRecommended,
    },
    targetEncoding: {
      isFetching: isFetchingTargetEncoding,
      error: targetEncodingError,
      suggestions: targetEncodingSuggestions,
      metadata: targetEncodingMetadata,
      handleApplyRecommended: handleApplyTargetEncodingRecommended,
    },
    hashEncoding: {
      isFetching: isFetchingHashEncoding,
      error: hashEncodingError,
      suggestions: hashEncodingSuggestions,
      metadata: hashEncodingMetadata,
      handleApplyRecommended: handleApplyHashEncodingRecommended,
    },
    ordinalEncoding: {
      isFetching: isFetchingOrdinalEncoding,
      error: ordinalEncodingError,
      suggestions: ordinalEncodingSuggestions,
      metadata: ordinalEncodingMetadata,
      handleApplyRecommended: handleApplyOrdinalEncodingRecommended,
    },
    dummyEncoding: {
      isFetching: isFetchingDummyEncoding,
      error: dummyEncodingError,
      suggestions: dummyEncodingSuggestions,
      metadata: dummyEncodingMetadata,
      handleApplyRecommended: handleApplyDummyEncodingRecommended,
    },
    oneHotEncoding: {
      isFetching: isFetchingOneHotEncoding,
      error: oneHotEncodingError,
      suggestions: oneHotEncodingSuggestions,
      metadata: oneHotEncodingMetadata,
      handleApplyRecommended: handleApplyOneHotEncodingRecommended,
    },
  };
};
