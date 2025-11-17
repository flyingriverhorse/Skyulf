// @ts-nocheck
import { BinnedDistributionResponse, FetchBinnedDistributionOptions } from '../types';

export async function fetchBinnedDistribution(
  sourceId: string,
  options: FetchBinnedDistributionOptions = {}
): Promise<BinnedDistributionResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    sampleSize === 'all'
      ? 0
      : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
        ? Math.max(0, Math.round(sampleSize))
        : 500;

  const body: Record<string, any> = {
    dataset_source_id: sourceId,
    sample_size: normalizedSampleSize,
  };

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    body.graph = { nodes: graph.nodes, edges: graph.edges };
  }

  if (targetNodeId) {
    body.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/analytics/binned-distribution', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = 'Failed to load binned column distributions';
    try {
      const errorPayload = await response.clone().json();
      const detail =
        (typeof errorPayload?.detail === 'string' && errorPayload.detail.trim())
          ? errorPayload.detail.trim()
          : (typeof errorPayload?.error === 'string' && errorPayload.error.trim())
            ? errorPayload.error.trim()
            : (typeof errorPayload?.message === 'string' && errorPayload.message.trim())
              ? errorPayload.message.trim()
              : null;
      if (detail) {
        message = detail;
      }
    } catch {
      try {
        const fallbackText = await response.text();
        const trimmed = fallbackText.trim();
        if (trimmed) {
          message = trimmed;
        }
      } catch {
        // Ignore secondary parsing failures
      }
    }

    throw new Error(message);
  }

  return response.json();
}
