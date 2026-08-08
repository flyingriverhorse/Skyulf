import { apiClient } from './client';

export interface DeploymentInfo {
  id: number;
  job_id: string;
  model_type: string;
  artifact_uri: string;
  is_active: boolean;
  created_at: string;
  input_schema?: { name: string; type: string }[];
  target_column?: string;
  /** Dataset backing the deployed job, when the source training job is still resolvable. */
  dataset_id?: string | null;
  /** Shared version sequence for this model_type/dataset, matching the Registry entry. */
  version?: number | string | null;
  /** The deployment this one replaced, so History can render a replacement chain. */
  previous_deployment_id?: number | null;
}

export interface PredictionRequest {
  data: unknown[];
  override_thresholds?: Record<string, number> | null;
}

export interface PredictionResponse {
  predictions: unknown[];
  model_version: string;
  thresholds_applied?: Record<string, number> | null;
}

export const deploymentApi = {
  deployModel: async (jobId: string): Promise<DeploymentInfo> => {
    const response = await apiClient.post<DeploymentInfo>(`/deployment/deploy/${jobId}`);
    return response.data;
  },

  getActive: async (): Promise<DeploymentInfo | null> => {
    try {
      const response = await apiClient.get<DeploymentInfo>('/deployment/active');
      return response.data;
    } catch (error: unknown) {
      const err = error as { response?: { status: number } };
      if (err.response && err.response.status === 404) {
        return null;
      }
      throw error;
    }
  },

  getHistory: async (limit: number = 50, skip: number = 0): Promise<DeploymentInfo[]> => {
    const response = await apiClient.get<DeploymentInfo[]>('/deployment/history', { params: { limit, skip } });
    return response.data;
  },

  deactivate: async (): Promise<void> => {
    await apiClient.post('/deployment/deactivate');
  },

  predict: async (
    data: unknown[],
    overrideThresholds?: Record<string, number> | null,
    options?: { signal?: AbortSignal },
  ): Promise<PredictionResponse> => {
    const response = await apiClient.post<PredictionResponse>(
      '/deployment/predict',
      {
        data,
        override_thresholds: overrideThresholds,
      },
      options?.signal ? { signal: options.signal } : undefined,
    );
    return response.data;
  }
};
