import { useMemo } from 'react';
import { useGraphStore } from '../store/useGraphStore';
import { useUpstreamData } from './useUpstreamData';

interface UseRecommendationsOptions {
  /**
   * Filter by recommendation category (e.g., 'cleaning', 'feature_selection').
   * Matches if the recommendation's type is in this list.
   */
  types?: string[];

  /**
   * Filter by specific suggested node types (e.g., 'DropMissingColumns').
   * Matches if the recommendation's suggested_node_type is in this list.
   */
  suggestedNodeTypes?: string[];

  /**
   * Filter by scope.
   * 'column': Requires target_columns to be present and non-empty.
   * 'row': (Future) Could require row indices or specific row-level metadata.
   * 'any': No scope filtering.
   */
  scope?: 'column' | 'any';
}

/**
 * A hook to retrieve and filter recommendations relevant to a specific node.
 * It handles checking upstream data connectivity and execution results.
 */
export const useRecommendations = (nodeId: string, options: UseRecommendationsOptions = {}) => {
  const upstreamData = useUpstreamData(nodeId);
  const datasetId = upstreamData.find(d => d.datasetId)?.datasetId;
  const executionResult = useGraphStore(state => state.executionResult);

  return useMemo(() => {
    // 1. Basic Availability Check
    if (!datasetId || !executionResult?.recommendations) {
      return [];
    }

    return executionResult.recommendations.filter(r => {
      // 2. Match Type OR Suggested Node Type
      // If neither is provided, we assume we want all recommendations (subject to scope).
      // If provided, at least one must match.
      if (options.types || options.suggestedNodeTypes) {
        const typeMatch = options.types?.includes(r.type);
        const nodeTypeMatch = options.suggestedNodeTypes?.includes(r.suggested_node_type);
        
        if (!typeMatch && !nodeTypeMatch) {
          return false;
        }
      }

      // 3. Scope Filtering
      if (options.scope === 'column') {
        if (!r.target_columns || r.target_columns.length === 0) {
          return false;
        }
      }

      return true;
    });
  }, [executionResult, datasetId, options.types, options.suggestedNodeTypes, options.scope]);
};
