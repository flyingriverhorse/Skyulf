// @ts-nocheck
import {
  SkewnessRecommendationsResponse,
  FetchSkewnessRecommendationsOptions,
  ScalingRecommendationsResponse,
  FetchScalingRecommendationsOptions,
  BinningRecommendationsResponse,
  FetchBinningRecommendationsOptions,
} from '../types';

export async function fetchSkewnessRecommendations(
  sourceId: string,
  options: FetchSkewnessRecommendationsOptions = {}
): Promise<SkewnessRecommendationsResponse> {
  const { sampleSize = 500, transformations, graph, targetNodeId } = options;

  const body: any = {
    dataset_source_id: sourceId,
    sample_size: sampleSize,
  };

  if (Array.isArray(transformations)) {
    if (transformations.length > 0) {
      body.transformations = JSON.stringify(transformations);
    }
  } else if (transformations && typeof transformations === 'object') {
    const keys = Object.keys(transformations);
    if (keys.length > 0) {
      body.transformations = JSON.stringify(transformations);
    }
  }

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/recommendations/skewness', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load skewness recommendations');
  }

  return response.json();
}

export async function fetchScalingRecommendations(
  sourceId: string,
  options: FetchScalingRecommendationsOptions = {}
): Promise<ScalingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/scaling', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load scaling recommendations');
  }

  return response.json();
}

export async function fetchBinningRecommendations(
  sourceId: string,
  options: FetchBinningRecommendationsOptions = {}
): Promise<BinningRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/binning', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load binning recommendations');
  }

  return response.json();
}
