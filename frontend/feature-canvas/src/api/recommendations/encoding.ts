// @ts-nocheck
import {
  LabelEncodingRecommendationsResponse,
  FetchLabelEncodingRecommendationsOptions,
  OrdinalEncodingRecommendationsResponse,
  FetchOrdinalEncodingRecommendationsOptions,
  TargetEncodingRecommendationsResponse,
  FetchTargetEncodingRecommendationsOptions,
  HashEncodingRecommendationsResponse,
  FetchHashEncodingRecommendationsOptions,
  OneHotEncodingRecommendationsResponse,
  FetchOneHotEncodingRecommendationsOptions,
  DummyEncodingRecommendationsResponse,
  FetchDummyEncodingRecommendationsOptions,
} from '../types';

export async function fetchLabelEncodingRecommendations(
  sourceId: string,
  options: FetchLabelEncodingRecommendationsOptions = {}
): Promise<LabelEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/label-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load label encoding recommendations');
  }

  return response.json();
}

export async function fetchOrdinalEncodingRecommendations(
  sourceId: string,
  options: FetchOrdinalEncodingRecommendationsOptions = {}
): Promise<OrdinalEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/ordinal-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load ordinal encoding suggestions');
  }

  return response.json();
}

export async function fetchTargetEncodingRecommendations(
  sourceId: string,
  options: FetchTargetEncodingRecommendationsOptions = {}
): Promise<TargetEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/target-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load target encoding recommendations');
  }

  return response.json();
}

export async function fetchHashEncodingRecommendations(
  sourceId: string,
  options: FetchHashEncodingRecommendationsOptions = {}
): Promise<HashEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/hash-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load hash encoding recommendations');
  }

  return response.json();
}

export async function fetchOneHotEncodingRecommendations(
  sourceId: string,
  options: FetchOneHotEncodingRecommendationsOptions = {}
): Promise<OneHotEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/one-hot-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load one-hot encoding recommendations');
  }

  return response.json();
}

export async function fetchDummyEncodingRecommendations(
  sourceId: string,
  options: FetchDummyEncodingRecommendationsOptions = {}
): Promise<DummyEncodingRecommendationsResponse> {
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

  const response = await fetch('/ml-workflow/api/recommendations/dummy-encoding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('Failed to load dummy encoding recommendations');
  }

  return response.json();
}
