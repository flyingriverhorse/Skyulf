// @ts-nocheck
import {
  FetchPipelinePreviewRowsOptions,
  PipelinePreviewRowsResponse,
  PipelinePreviewRequest,
  PipelinePreviewResponse,
  FullExecutionSignal,
} from '../types';

export async function fetchPipelinePreviewRows(
  datasetSourceId: string,
  options: FetchPipelinePreviewRowsOptions = {}
): Promise<PipelinePreviewRowsResponse> {
  const { offset, limit, mode } = options;

  const params = new URLSearchParams();

  if (typeof offset === 'number' && Number.isFinite(offset)) {
    params.set('offset', String(offset));
  }

  if (typeof limit === 'number' && Number.isFinite(limit)) {
    params.set('limit', String(limit));
  }

  if (mode) {
    params.set('mode', mode);
  }

  const query = params.toString();
  const url = `/ml-workflow/api/pipelines/${encodeURIComponent(datasetSourceId)}/preview/rows${
    query ? `?${query}` : ''
  }`;

  const response = await fetch(url);

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load preview rows');
  }

  return response.json();
}

export async function fetchPipelinePreview(
  payload: PipelinePreviewRequest
): Promise<PipelinePreviewResponse> {
  const response = await fetch('/ml-workflow/api/pipelines/preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load pipeline preview');
  }

  return response.json();
}

export async function triggerFullDatasetExecution(
  payload: PipelinePreviewRequest
): Promise<PipelinePreviewResponse> {
  const fullDatasetPayload = {
    ...payload,
    sample_size: 0,
  };

  const response = await fetch('/ml-workflow/api/pipelines/preview', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(fullDatasetPayload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to trigger full dataset execution');
  }

  return response.json();
}

export async function fetchFullExecutionStatus(
  datasetSourceId: string,
  jobId: string
): Promise<FullExecutionSignal> {
  const response = await fetch(
    `/ml-workflow/api/pipelines/${encodeURIComponent(datasetSourceId)}/full-execution/${encodeURIComponent(jobId)}`
  );

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch full execution status');
  }

  return response.json();
}
