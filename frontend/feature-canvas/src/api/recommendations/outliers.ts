// @ts-nocheck
import {
  DropColumnRecommendationsResponse,
  FetchDropColumnRecommendationsOptions,
  OutlierRecommendationsResponse,
  FetchOutlierRecommendationsOptions,
} from '../types';

export async function fetchDropColumnRecommendations(
  sourceId: string,
  options: FetchDropColumnRecommendationsOptions = {}
): Promise<DropColumnRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/drop-columns', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load column drop recommendations');
  }

  return response.json();
}

export async function fetchOutlierRecommendations(
  sourceId: string,
  options: FetchOutlierRecommendationsOptions = {}
): Promise<OutlierRecommendationsResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    typeof sampleSize === 'number' && Number.isFinite(sampleSize)
      ? Math.min(5000, Math.max(50, Math.round(sampleSize)))
      : 500;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: normalizedSampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/outliers', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load outlier recommendations');
  }

  return response.json();
}
