import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchOrdinalEncodingRecommendations,
  type FeatureGraph,
  type OrdinalEncodingColumnSuggestion,
  type OrdinalEncodingRecommendationsResponse,
  type OrdinalEncodingSuggestionStatus,
} from '../../../api';
import { type CatalogFlagMap } from './useCatalogFlags';

type OrdinalEncodingRecommendationMetadata = {
  sampleSize: number | null;
  generatedAt: string | null;
  totalTextColumns: number;
  recommendedCount: number;
  autoDetectDefault: boolean;
  enableUnknownDefault: boolean;
  highCardinalityColumns: string[];
  notes: string[];
};

type UseOrdinalEncodingRecommendationsArgs = {
  catalogFlags: CatalogFlagMap;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext?: FeatureGraph | { nodes: any[]; edges: any[] } | null;
  targetNodeId?: string | null;
};

type UseOrdinalEncodingRecommendationsResult = {
  isFetching: boolean;
  error: string | null;
  suggestions: OrdinalEncodingColumnSuggestion[];
  metadata: OrdinalEncodingRecommendationMetadata;
  refresh: () => void;
};

const DEFAULT_METADATA: OrdinalEncodingRecommendationMetadata = {
  sampleSize: null,
  generatedAt: null,
  totalTextColumns: 0,
  recommendedCount: 0,
  autoDetectDefault: false,
  enableUnknownDefault: false,
  highCardinalityColumns: [],
  notes: [],
};

const normalizeStatus = (value: unknown): OrdinalEncodingSuggestionStatus => {
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
  payload?: OrdinalEncodingRecommendationsResponse | null,
): OrdinalEncodingColumnSuggestion[] => {
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
      const recommendedHandleUnknown = typeof entry.recommended_handle_unknown === 'boolean'
        ? entry.recommended_handle_unknown
        : Boolean(entry.missing_percentage && entry.missing_percentage > 0);

      const sanitized: OrdinalEncodingColumnSuggestion = {
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
        recommended_handle_unknown: recommendedHandleUnknown,
      };

      return sanitized;
    })
    .filter((entry): entry is OrdinalEncodingColumnSuggestion => Boolean(entry));
};

const extractMetadata = (
  payload?: OrdinalEncodingRecommendationsResponse | null,
): OrdinalEncodingRecommendationMetadata => {
  if (!payload || typeof payload !== 'object') {
    return { ...DEFAULT_METADATA };
  }

  const sampleSize = typeof payload.sample_size === 'number' ? payload.sample_size : null;
  const generatedAt = typeof payload.generated_at === 'string' ? payload.generated_at : null;
  const totalTextColumns = Number(payload.total_text_columns ?? 0) || 0;
  const recommendedCount = Number(payload.recommended_count ?? 0) || 0;
  const enableUnknownDefault = Boolean(payload.enable_unknown_default);
  const highCardinalityColumns = coerceStringArray(payload.high_cardinality_columns);
  const notes = coerceStringArray(payload.notes);
  const autoDetectDefault = Boolean(payload.auto_detect_default);

  return {
    sampleSize,
    generatedAt,
    totalTextColumns,
    recommendedCount,
    autoDetectDefault,
    enableUnknownDefault,
    highCardinalityColumns,
    notes,
  };
};

export const useOrdinalEncodingRecommendations = ({
  catalogFlags,
  sourceId,
  hasReachableSource,
  graphContext,
  targetNodeId,
}: UseOrdinalEncodingRecommendationsArgs): UseOrdinalEncodingRecommendationsResult => {
  const { isOrdinalEncodingNode } = catalogFlags;
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<OrdinalEncodingColumnSuggestion[]>([]);
  const [metadata, setMetadata] = useState<OrdinalEncodingRecommendationMetadata>({ ...DEFAULT_METADATA });
  const [requestId, setRequestId] = useState(0);

  useEffect(() => {
    let isActive = true;

    const resetState = (message?: string | null) => {
      setSuggestions([]);
      setMetadata({ ...DEFAULT_METADATA });
      setIsFetching(false);
      setError(message ?? null);
    };

    if (!isOrdinalEncodingNode) {
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

    fetchOrdinalEncodingRecommendations(sourceId, {
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
        const message = typeof err?.message === 'string' ? err.message : 'Unable to load ordinal encoding suggestions';
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
  }, [graphContext, hasReachableSource, isOrdinalEncodingNode, requestId, sourceId, targetNodeId]);

  const refresh = useCallback(() => {
    if (!isOrdinalEncodingNode || !sourceId || !hasReachableSource) {
      return;
    }
    setRequestId((value) => value + 1);
  }, [hasReachableSource, isOrdinalEncodingNode, sourceId]);

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
