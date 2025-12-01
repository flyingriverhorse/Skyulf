// @ts-nocheck
import {
  HyperparameterTuningJobCreatePayload,
  HyperparameterTuningJobBatchResponse,
  HyperparameterTuningJobResponse,
  HyperparameterTuningJobListResponse,
  FetchHyperparameterTuningJobsOptions,
  ModelHyperparametersResponse,
  BestHyperparametersResponse,
  FetchBestHyperparametersOptions,
} from '../types';

export async function createHyperparameterTuningJob(
  payload: HyperparameterTuningJobCreatePayload
): Promise<HyperparameterTuningJobBatchResponse> {
  const response = await fetch('/ml-workflow/api/hyperparameter-tuning-jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to launch tuning jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to enqueue tuning job');
  }

  const data = await response.json();

  if (data && Array.isArray(data?.jobs)) {
    const jobs: HyperparameterTuningJobResponse[] = data.jobs;
    const total = typeof data.total === 'number' ? data.total : jobs.length;
    return { jobs, total };
  }

  if (data && data.id) {
    return { jobs: [data as HyperparameterTuningJobResponse], total: 1 };
  }

  return { jobs: [], total: 0 };
}

export async function cancelHyperparameterTuningJob(jobId: string): Promise<void> {
  if (!jobId) {
    throw new Error('jobId is required');
  }

  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning-jobs/${encodeURIComponent(jobId)}/cancel`,
    {
      method: 'POST',
      credentials: 'include',
    }
  );

  if (!response.ok) {
    const detail = await response.text();
    // Try to parse JSON error detail if possible
    let errorMessage = detail;
    try {
      const json = JSON.parse(detail);
      if (json.detail) errorMessage = json.detail;
    } catch (e) {
      // ignore
    }

    if (response.status === 501) {
      throw new Error(errorMessage || 'Cancellation is not supported in this environment.');
    }
    throw new Error(errorMessage || 'Failed to cancel tuning job');
  }
}

export async function fetchHyperparameterTuningJob(
  jobId: string
): Promise<HyperparameterTuningJobResponse> {
  if (!jobId) {
    throw new Error('jobId is required');
  }

  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning-jobs/${encodeURIComponent(jobId)}`,
    {
      credentials: 'include',
    }
  );

  if (response.status === 404) {
    throw new Error('Tuning job not found');
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view this tuning job.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load tuning job');
  }

  return response.json();
}

export async function fetchHyperparameterTuningJobs(
  options: FetchHyperparameterTuningJobsOptions = {}
): Promise<HyperparameterTuningJobListResponse> {
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
  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning-jobs${query ? `?${query}` : ''}`,
    {
      credentials: 'include',
    }
  );

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to view tuning jobs.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load tuning jobs');
  }

  return response.json();
}

export async function fetchModelHyperparameters(
  modelType: string
): Promise<ModelHyperparametersResponse> {
  const response = await fetch(`/ml-workflow/api/model-hyperparameters/${encodeURIComponent(modelType)}`);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model type '${modelType}' not found`);
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to load model hyperparameters');
  }

  return response.json();
}

export async function fetchBestHyperparameters(
  modelType: string,
  options: FetchBestHyperparametersOptions = {}
): Promise<BestHyperparametersResponse> {
  const params = new URLSearchParams();

  if (options.pipelineId) {
    params.set('pipeline_id', options.pipelineId);
  }
  if (options.datasetSourceId) {
    params.set('dataset_source_id', options.datasetSourceId);
  }

  const query = params.toString();
  const response = await fetch(
    `/ml-workflow/api/hyperparameter-tuning/best-params/${encodeURIComponent(modelType)}${
      query ? `?${query}` : ''
    }`,
    {
      credentials: 'include',
    }
  );

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model type '${modelType}' not found`);
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch best hyperparameters');
  }

  return response.json();
}
