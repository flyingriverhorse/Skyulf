import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchOneHotEncodingRecommendations,
  type FeatureGraph,
  type OneHotEncodingColumnSuggestion,
  type OneHotEncodingRecommendationsResponse,
  type OneHotEncodingSuggestionStatus,
} from '../../../api';

type OneHotEncodingRecommendationMetadata = {
  sampleSize: number | null;
  generatedAt: string | null;
  totalTextColumns: number;
  recommendedCount: number;
  cautionedCount: number;
  highCardinalityColumns: string[];
  notes: string[];
};

type UseOneHotEncodingRecommendationsArgs = {
  isOneHotEncodingNode: boolean;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

type UseOneHotEncodingRecommendationsResult = {
  isFetching: boolean;
  error: string | null;
  suggestions: OneHotEncodingColumnSuggestion[];
  metadata: OneHotEncodingRecommendationMetadata;
  refresh: () => void;
};

const DEFAULT_METADATA: OneHotEncodingRecommendationMetadata = {
  sampleSize: null,
  generatedAt: null,
  totalTextColumns: 0,
  recommendedCount: 0,
  cautionedCount: 0,
  highCardinalityColumns: [],
  notes: [],
};

const normalizeStatus = (value: unknown): OneHotEncodingSuggestionStatus => {
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
  payload?: OneHotEncodingRecommendationsResponse | null,
): OneHotEncodingColumnSuggestion[] => {
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
      const sampleValues = coerceStringArray((entry as any).sample_values);
      const estimatedDummyColumns = typeof entry.estimated_dummy_columns === 'number'
        ? Math.max(0, Math.round(entry.estimated_dummy_columns))
        : 0;
      const score = typeof entry.score === 'number' ? entry.score : 0;
      const selectable = typeof entry.selectable === 'boolean' ? entry.selectable : true;
      const recommendedDropFirst = typeof entry.recommended_drop_first === 'boolean'
        ? entry.recommended_drop_first
        : false;

      const sanitized: OneHotEncodingColumnSuggestion = {
        column,
        status,
        reason,
        dtype,
        unique_count: uniqueCount,
        unique_percentage: uniquePercentage,
        missing_percentage: missingPercentage,
        text_category: textCategory,
        sample_values: sampleValues,
        estimated_dummy_columns: estimatedDummyColumns,
        score,
        selectable,
        recommended_drop_first: recommendedDropFirst,
      };

      return sanitized;
    })
    .filter((entry): entry is OneHotEncodingColumnSuggestion => Boolean(entry));
};

const extractMetadata = (
  payload?: OneHotEncodingRecommendationsResponse | null,
): OneHotEncodingRecommendationMetadata => {
  if (!payload || typeof payload !== 'object') {
    return { ...DEFAULT_METADATA };
  }

  const sampleSize = typeof payload.sample_size === 'number' ? payload.sample_size : null;
  const generatedAt = typeof payload.generated_at === 'string' ? payload.generated_at : null;
  const totalTextColumns = Number(payload.total_text_columns ?? 0) || 0;
  const recommendedCount = Number(payload.recommended_count ?? 0) || 0;
  const cautionedCount = Number(payload.cautioned_count ?? 0) || 0;
  const highCardinalityColumns = coerceStringArray(payload.high_cardinality_columns);
  const notes = coerceStringArray(payload.notes);

  return {
    sampleSize,
    generatedAt,
    totalTextColumns,
    recommendedCount,
    cautionedCount,
    highCardinalityColumns,
    notes,
  };
};

export const useOneHotEncodingRecommendations = ({
  isOneHotEncodingNode,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
}: UseOneHotEncodingRecommendationsArgs): UseOneHotEncodingRecommendationsResult => {
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<OneHotEncodingColumnSuggestion[]>([]);
  const [metadata, setMetadata] = useState<OneHotEncodingRecommendationMetadata>({ ...DEFAULT_METADATA });
  const [requestId, setRequestId] = useState(0);

  useEffect(() => {
    let isActive = true;

    const resetState = (message?: string | null) => {
      setSuggestions([]);
      setMetadata({ ...DEFAULT_METADATA });
      setIsFetching(false);
      setError(message ?? null);
    };

    if (!isOneHotEncodingNode) {
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

    fetchOneHotEncodingRecommendations(sourceId, {
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
        const message = typeof err?.message === 'string' ? err.message : 'Unable to load one-hot encoding suggestions';
        resetState(message);
      })
      .finally(() => {
        if (isActive) {
          setIsFetching(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [graphContext, hasReachableSource, isOneHotEncodingNode, requestId, sourceId, targetNodeId]);

  const refresh = useCallback(() => {
    if (!isOneHotEncodingNode || !sourceId || !hasReachableSource) {
      return;
    }
    setRequestId((value) => value + 1);
  }, [hasReachableSource, isOneHotEncodingNode, sourceId]);

  const memoizedMetadata = useMemo(() => metadata, [metadata]);
  const memoizedSuggestions = useMemo(() => suggestions, [suggestions]);

  return {
    isFetching,
    error,
    suggestions: memoizedSuggestions,
    metadata: memoizedMetadata,
    refresh,
  };
};
