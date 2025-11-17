// @ts-nocheck
import { FeaturePipelineResponse, FeaturePipelinePayload } from '../types';

export async function fetchPipeline(
  sourceId: string
): Promise<FeaturePipelineResponse | null> {
  const response = await fetch(`/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}`);

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    throw new Error('Failed to load saved pipeline');
  }

  return response.json();
}

export async function savePipeline(
  sourceId: string,
  payload: FeaturePipelinePayload
): Promise<FeaturePipelineResponse> {
  const response = await fetch(`/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Failed to save pipeline');
  }

  return response.json();
}

export async function fetchPipelineHistory(
  sourceId: string,
  limit = 10
): Promise<FeaturePipelineResponse[]> {
  const response = await fetch(
    `/ml-workflow/api/pipelines/${encodeURIComponent(sourceId)}/history?limit=${encodeURIComponent(
      String(limit)
    )}`
  );

  if (response.status === 404) {
    return [];
  }

  if (!response.ok) {
    throw new Error('Failed to load pipeline history');
  }

  return response.json();
}
