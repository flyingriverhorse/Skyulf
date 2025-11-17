// @ts-nocheck
import { ModelEvaluationRequest, ModelEvaluationReport } from '../types';

export async function evaluateTrainingJob(
  jobId: string,
  payload: ModelEvaluationRequest
): Promise<ModelEvaluationReport> {
  if (!jobId) {
    throw new Error('jobId is required for evaluation');
  }

  const response = await fetch(
    `/ml-workflow/api/training-jobs/${encodeURIComponent(jobId)}/evaluate`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify(payload ?? {}),
    }
  );

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error('Sign in to evaluate this model.');
    }
    const detail = await response.text();
    throw new Error(detail || 'Failed to evaluate model');
  }

  return response.json();
}
