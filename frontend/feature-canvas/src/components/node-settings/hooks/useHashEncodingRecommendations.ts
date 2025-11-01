import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchHashEncodingRecommendations,
  type FeatureGraph,
  type HashEncodingColumnSuggestion,
  type HashEncodingRecommendationsResponse,
  type HashEncodingSuggestionStatus,
} from '../../../api';

type HashEncodingRecommendationMetadata = {
  sampleSize: number | null;
  generatedAt: string | null;
  totalTextColumns: number;
  recommendedCount: number;
  autoDetectDefault: boolean;
  suggestedBucketDefault: number;
  highCardinalityColumns: string[];
  notes: string[];
};

type UseHashEncodingRecommendationsArgs = {
  isHashEncodingNode: boolean;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

type UseHashEncodingRecommendationsResult = {
  isFetching: boolean;
  error: string | null;
  suggestions: HashEncodingColumnSuggestion[];
  metadata: HashEncodingRecommendationMetadata;
  refresh: () => void;
};

const DEFAULT_METADATA: HashEncodingRecommendationMetadata = {
  sampleSize: null,
  generatedAt: null,
  totalTextColumns: 0,
  recommendedCount: 0,
  autoDetectDefault: false,
  suggestedBucketDefault: 0,
  highCardinalityColumns: [],
  notes: [],
};

const normalizeStatus = (value: unknown): HashEncodingSuggestionStatus => {
  if (typeof value !== 'string') {
    return 'recommended';
  }
  const normalized = value.trim().toLowerCase();
  switch (normalized) {
    case 'recommended':
    case 'high_cardinality':
    case 'identifier':
    case 'free_text':
    case 'single_category':
    case 'too_many_categories':
      return normalized;
    default:
      return 'recommended';
  }
};

const coerceStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => {
      if (entry === null || entry === undefined) {
        return '';
      }
      return String(entry);
    })
    .filter((entry) => entry.trim().length > 0);
};

const sanitizeSuggestions = (
  payload?: HashEncodingRecommendationsResponse | null,
): HashEncodingColumnSuggestion[] => {
  if (!payload || !Array.isArray(payload.columns)) {
    return [];
  }

  return payload.columns
    .map((entry) => {
      if (!entry || typeof entry !== 'object') {
        return null;
      }

      const column = typeof entry.column === 'string' ? entry.column.trim() : '';
      if (!column) {
        return null;
      }

      const status = normalizeStatus((entry as any).status);
      const reason = typeof entry.reason === 'string' ? entry.reason : '';
      const dtype = typeof entry.dtype === 'string' ? entry.dtype : null;
      const uniqueCount = typeof entry.unique_count === 'number' ? entry.unique_count : null;
      const uniquePercentage = typeof entry.unique_percentage === 'number' ? entry.unique_percentage : null;
      const missingPercentage = typeof entry.missing_percentage === 'number' ? entry.missing_percentage : null;
      const textCategory = typeof entry.text_category === 'string' ? entry.text_category : null;
      const sampleValues = coerceStringArray(entry.sample_values);
      const score = typeof entry.score === 'number' ? entry.score : 0;
      const selectable = typeof entry.selectable === 'boolean' ? entry.selectable : true;
      const recommendedBucketCount = typeof entry.recommended_bucket_count === 'number'
        ? Math.max(2, Math.round(entry.recommended_bucket_count))
        : 0;

      const sanitized: HashEncodingColumnSuggestion = {
        column,
        status,
        reason,
        dtype,
        unique_count: uniqueCount,
        unique_percentage: uniquePercentage,
        missing_percentage: missingPercentage,
        text_category: textCategory,
        sample_values: sampleValues,
        score,
        selectable,
        recommended_bucket_count: recommendedBucketCount,
      };

      return sanitized;
    })
    .filter((entry): entry is HashEncodingColumnSuggestion => Boolean(entry));
};

const extractMetadata = (
  payload?: HashEncodingRecommendationsResponse | null,
): HashEncodingRecommendationMetadata => {
  if (!payload || typeof payload !== 'object') {
    return { ...DEFAULT_METADATA };
  }

  const sampleSize = typeof payload.sample_size === 'number' ? payload.sample_size : null;
  const generatedAt = typeof payload.generated_at === 'string' ? payload.generated_at : null;
  const totalTextColumns = Number(payload.total_text_columns ?? 0) || 0;
  const recommendedCount = Number(payload.recommended_count ?? 0) || 0;
  const autoDetectDefault = Boolean(payload.auto_detect_default);
  const suggestedBucketDefault = Number(payload.suggested_bucket_default ?? 0) || 0;
  const highCardinalityColumns = coerceStringArray(payload.high_cardinality_columns);
  const notes = coerceStringArray(payload.notes);

  return {
    sampleSize,
    generatedAt,
    totalTextColumns,
    recommendedCount,
    autoDetectDefault,
    suggestedBucketDefault,
    highCardinalityColumns,
    notes,
  };
};

export const useHashEncodingRecommendations = ({
  isHashEncodingNode,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
}: UseHashEncodingRecommendationsArgs): UseHashEncodingRecommendationsResult => {
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<HashEncodingColumnSuggestion[]>([]);
  const [metadata, setMetadata] = useState<HashEncodingRecommendationMetadata>({ ...DEFAULT_METADATA });
  const [requestId, setRequestId] = useState(0);

  const resetState = useCallback((message?: string | null) => {
    setSuggestions([]);
    setMetadata({ ...DEFAULT_METADATA });
    setIsFetching(false);
    setError(message ?? null);
  }, []);

  useEffect(() => {
    let isActive = true;

    if (!isHashEncodingNode) {
      resetState();
      return () => {
        isActive = false;
      };
    }

    if (!sourceId) {
      resetState();
      return () => {
        isActive = false;
      };
    }

    if (!hasReachableSource) {
      resetState('Connect this node to the dataset input to load suggestions.');
      return () => {
        isActive = false;
      };
    }

    setIsFetching(true);
    setError(null);

    fetchHashEncodingRecommendations(sourceId, {
      graph: graphContext ?? null,
      targetNodeId: targetNodeId ?? null,
    })
      .then((result) => {
        if (!isActive) {
          return;
        }
        setSuggestions(sanitizeSuggestions(result));
        setMetadata(extractMetadata(result));
      })
      .catch((err: any) => {
        if (!isActive) {
          return;
        }
        setSuggestions([]);
        setMetadata({ ...DEFAULT_METADATA });
        setError(err?.message ? String(err.message) : 'Unable to load hash encoding suggestions');
      })
      .finally(() => {
        if (isActive) {
          setIsFetching(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [graphContext, hasReachableSource, isHashEncodingNode, resetState, sourceId, targetNodeId, requestId]);

  const refresh = useCallback(() => {
    setRequestId((previous) => previous + 1);
  }, []);

  const value = useMemo<UseHashEncodingRecommendationsResult>(() => ({
    isFetching,
    error,
    suggestions,
    metadata,
    refresh,
  }), [error, isFetching, metadata, refresh, suggestions]);

  return value;
};
