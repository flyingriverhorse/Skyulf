// @ts-nocheck
import { QuickProfileResponse, FetchQuickProfileOptions } from '../types';

export async function fetchQuickProfile(
  sourceId: string,
  options: FetchQuickProfileOptions = {}
): Promise<QuickProfileResponse> {
  const { sampleSize = 500, graph, targetNodeId } = options;

  const normalizedSampleSize =
    sampleSize === 'all'
      ? 0
      : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
        ? Math.max(0, Math.round(sampleSize))
        : 500;

  const params = new URLSearchParams({
    dataset_source_id: sourceId,
    sample_size: String(normalizedSampleSize),
  });

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    params.set('graph', JSON.stringify({ nodes: graph.nodes, edges: graph.edges }));
  }

  if (targetNodeId) {
    params.set('target_node_id', targetNodeId);
  }

  const response = await fetch(`/ml-workflow/api/analytics/quick-profile?${params.toString()}`);

  if (!response.ok) {
    let message = 'Failed to generate dataset profile';
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
        /* swallow - retain default message */
      }
    }
    throw new Error(message);
  }

  return response.json();
}
