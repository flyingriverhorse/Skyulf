// @ts-nocheck
import {
  TrainingJobCreatePayload,
  TrainingJobBatchResponse,
  TrainingJobResponse,
  TrainingJobListResponse,
  FetchTrainingJobsOptions,
} from '../types';

export async function createTrainingJob(
  payload: TrainingJobCreatePayload
): Promise<TrainingJobBatchResponse> {
  const response = await fetch('/ml-workflow/api/training-jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to launch training jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to enqueue training job');
  }

  const data = await response.json();

  if (data && Array.isArray(data?.jobs)) {
    const jobs: TrainingJobResponse[] = data.jobs;
    const total = typeof data.total === 'number' ? data.total : jobs.length;
    return { jobs, total };
  }

  if (data && data.id) {
    return { jobs: [data as TrainingJobResponse], total: 1 };
  }

  return { jobs: [], total: 0 };
}

export async function fetchTrainingJob(jobId: string): Promise<TrainingJobResponse> {
  if (!jobId) {
    throw new Error('jobId is required');
  }

  const response = await fetch(`/ml-workflow/api/training-jobs/${encodeURIComponent(jobId)}`, {
    credentials: 'include',
  });

  if (response.status === 404) {
    throw new Error('Training job not found');
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view this training job.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load training job');
  }

  return response.json();
}

export async function fetchTrainingJobs(
  options: FetchTrainingJobsOptions = {}
): Promise<TrainingJobListResponse> {
  const params = new URLSearchParams();

  if (options.datasetSourceId) {
    params.set('dataset_source_id', options.datasetSourceId);
  }

  if (options.pipelineId) {
    params.set('pipeline_id', options.pipelineId);
  }

  if (options.nodeId) {
    params.set('node_id', options.nodeId);
  }

  if (typeof options.limit === 'number' && Number.isFinite(options.limit)) {
    params.set('limit', String(options.limit));
  }

  const query = params.toString();
  const response = await fetch(`/ml-workflow/api/training-jobs${query ? `?${query}` : ''}`, {
    credentials: 'include',
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view training jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load training jobs');
  }

  return response.json();
}
